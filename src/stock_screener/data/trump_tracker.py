"""Trump tracker: Truth Social + news + optional X → ticker scores."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from stock_screener.data.trump_lexicon import extract_matches
from stock_screener.data.trump_news import fetch_trump_news
from stock_screener.data.trump_truth import TrumpPost, fetch_truth_posts
from stock_screener.data.trump_wh import fetch_white_house
from stock_screener.data.trump_x import fetch_trump_x_posts

logger = logging.getLogger(__name__)

SOURCE_WEIGHT = {
    "truth_social": 1.0,
    "white_house": 0.9,
    "x": 0.85,
    "news": 0.65,
}


@dataclass
class TrumpTickerScore:
    ticker: str
    score: float
    mention_weight: float
    mention_count: int
    themes: str
    sources: str
    sample_text: str
    sample_url: str
    why: str
    pass_filters: bool
    filter_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _age_hours(published: datetime | None, now: datetime) -> float:
    if published is None:
        return 36.0  # unknown → mid freshness
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return max(0.0, (now - published.astimezone(timezone.utc)).total_seconds() / 3600.0)


def _freshness(hours: float, half_life_h: float = 36.0) -> float:
    # 1.0 at t=0, ~0.5 at half_life, floors at 0.15
    return max(0.15, math.exp(-math.log(2) * hours / half_life_h))


def collect_trump_posts(
    *,
    enable_truth: bool = True,
    enable_news: bool = True,
    enable_wh: bool = True,
    enable_x: bool = True,
    truth_max: int = 80,
) -> list[TrumpPost]:
    posts: list[TrumpPost] = []
    if enable_truth:
        posts.extend(fetch_truth_posts(max_items=truth_max))
    if enable_news:
        posts.extend(fetch_trump_news())
    if enable_wh:
        posts.extend(fetch_white_house())
    if enable_x:
        posts.extend(fetch_trump_x_posts())
    logger.info("Trump tracker collected %d posts/headlines", len(posts))
    return posts


def score_trump_tickers(
    posts: list[TrumpPost],
    *,
    min_mentions: int = 1,
) -> dict[str, TrumpTickerScore]:
    """Aggregate mention weights by ticker with source + freshness."""
    now = datetime.now(timezone.utc)
    weights: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    themes: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, tuple[float, str, str]] = {}

    for post in posts:
        matches = extract_matches(post.text)
        if not matches:
            continue
        age_h = _age_hours(post.published, now)
        fresh = _freshness(age_h)
        src_w = SOURCE_WEIGHT.get(post.source, 0.5)
        for hit in matches:
            w = hit.weight * src_w * fresh
            weights[hit.ticker] += w
            counts[hit.ticker] += 1
            themes[hit.ticker].add(hit.theme)
            sources[hit.ticker].add(post.source)
            prev = samples.get(hit.ticker)
            if prev is None or w > prev[0]:
                samples[hit.ticker] = (w, post.text[:180], post.url)

    if not weights:
        return {}

    max_w = max(weights.values()) or 1.0
    out: dict[str, TrumpTickerScore] = {}
    for ticker, w in weights.items():
        if counts[ticker] < min_mentions:
            continue
        # Map weight to 0–100 score (relative)
        score = round(35.0 + 65.0 * (w / max_w), 2)
        theme_s = ", ".join(sorted(themes[ticker])[:6])
        src_s = ", ".join(sorted(sources[ticker]))
        sample_w, sample_txt, sample_url = samples.get(ticker, (0.0, "", ""))
        why = (
            f"mentions={counts[ticker]}; weight={w:.2f}; themes={theme_s}; "
            f"sources={src_s}"
        )
        out[ticker] = TrumpTickerScore(
            ticker=ticker,
            score=score,
            mention_weight=round(w, 3),
            mention_count=counts[ticker],
            themes=theme_s,
            sources=src_s,
            sample_text=sample_txt,
            sample_url=sample_url,
            why=why,
            pass_filters=True,
            filter_reason="",
        )
    return out


def run_trump_tracker(
    *,
    enable_truth: bool = True,
    enable_news: bool = True,
    enable_wh: bool = True,
    enable_x: bool = True,
    truth_max: int = 80,
) -> tuple[list[TrumpPost], dict[str, TrumpTickerScore]]:
    posts = collect_trump_posts(
        enable_truth=enable_truth,
        enable_news=enable_news,
        enable_wh=enable_wh,
        enable_x=enable_x,
        truth_max=truth_max,
    )
    scores = score_trump_tickers(posts)
    logger.info("Trump tracker scored %d tickers", len(scores))
    return posts, scores
