"""Polymarket per-ticker odds for support/catalyst (free Gamma API)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

PM_SEARCH = "https://gamma-api.polymarket.com/public-search"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 "
    "Safari/537.36 stock-screener/0.1"
)

# Help search hit company-named markets for liquid mega-caps.
ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("NVIDIA", "Nvidia"),
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "GOOGL": ("Google", "Alphabet"),
    "GOOG": ("Google", "Alphabet"),
    "AMZN": ("Amazon",),
    "META": ("Meta", "Facebook"),
    "TSLA": ("Tesla",),
    "AMD": ("AMD", "Advanced Micro Devices"),
    "AVGO": ("Broadcom",),
    "NFLX": ("Netflix",),
    "CRM": ("Salesforce",),
    "ORCL": ("Oracle",),
    "INTC": ("Intel",),
}


@dataclass
class PolymarketAssessment:
    score: float  # 0–100 bullish tilt
    n_markets: int
    volume: float
    top_question: str
    reason: str

    @property
    def used(self) -> bool:
        return self.n_markets > 0


def _parse_json_field(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def _yes_prob(outcomes, prices) -> float | None:
    outcomes = _parse_json_field(outcomes) or []
    prices = _parse_json_field(prices) or []
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    try:
        probs = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() in {"yes", "up"}:
            return probs[i]
    return probs[0]


def _outcome_prob(outcomes, prices, needle: str) -> float | None:
    outcomes = _parse_json_field(outcomes) or []
    prices = _parse_json_field(prices) or []
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    needle_l = needle.lower()
    try:
        probs = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    for i, o in enumerate(outcomes):
        if needle_l in str(o).lower():
            return probs[i]
    return None


def _bullish_from_market(ticker: str, question: str, outcomes, prices) -> float | None:
    q = (question or "").lower()
    t = ticker.lower()
    aliases = (t,) + tuple(a.lower() for a in ALIASES.get(ticker.upper(), ()))
    if not any(a in q for a in aliases):
        named = any(_outcome_prob(outcomes, prices, a) is not None for a in aliases)
        if not named:
            return None

    for a in aliases:
        p = _outcome_prob(outcomes, prices, a)
        if p is not None and ("or" in q or "worth more" in q or "market cap" in q):
            return 100.0 * p

    yes = _yes_prob(outcomes, prices)
    if yes is None:
        return None

    if "up or down" in q or re.search(r"\bup\b.*\bdown\b", q):
        return 100.0 * yes
    if "(low)" in q or " hit (low)" in q or "close below" in q or "fall below" in q:
        return 100.0 * (1.0 - yes)
    if (
        "(high)" in q
        or " hit (high)" in q
        or "close above" in q
        or ("finish" in q and "above" in q)
        or "reach $" in q
        or "hit $" in q
    ):
        return 100.0 * yes
    if any(w in q for w in ("bankrupt", "delist", "fraud", "crash", "plunge")):
        return 100.0 * (1.0 - yes)
    return 50.0 + 40.0 * (yes - 0.5)


def fetch_polymarket_assessment(
    ticker: str,
    *,
    min_vol: float = 500.0,
    pause: float = 0.05,
    timeout: float = 15.0,
) -> PolymarketAssessment:
    """Volume-weighted bullish score from open Polymarket markets for ticker."""
    queries = [ticker] + list(ALIASES.get(ticker.upper(), ()))
    seen: set[str] = set()
    weighted: list[tuple[float, float]] = []
    top_q = ""
    top_vol = 0.0

    for q in queries:
        try:
            resp = requests.get(
                PM_SEARCH,
                params={"q": q},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            events = resp.json().get("events") or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("Polymarket search failed for %s (%s): %s", ticker, q, exc)
            continue
        for ev in events:
            for m in ev.get("markets") or []:
                if m.get("closed"):
                    continue
                mid = str(m.get("id") or m.get("conditionId") or m.get("question"))
                if mid in seen:
                    continue
                seen.add(mid)
                vol = float(m.get("volumeNum") or m.get("volume") or 0.0)
                if vol < min_vol:
                    continue
                question = m.get("question") or ev.get("title") or ""
                score = _bullish_from_market(
                    ticker, question, m.get("outcomes"), m.get("outcomePrices")
                )
                if score is None:
                    continue
                score = max(0.0, min(100.0, float(score)))
                weighted.append((score, vol))
                if vol > top_vol:
                    top_vol = vol
                    top_q = question[:140]
        if pause:
            time.sleep(pause)

    if not weighted:
        return PolymarketAssessment(
            score=50.0,
            n_markets=0,
            volume=0.0,
            top_question="",
            reason="Polymarket: no open markets",
        )

    vol_sum = sum(v for _, v in weighted)
    score = sum(s * v for s, v in weighted) / vol_sum
    short_q = top_q[:70] + ("…" if len(top_q) > 70 else "")
    return PolymarketAssessment(
        score=round(float(score), 2),
        n_markets=len(weighted),
        volume=float(vol_sum),
        top_question=top_q,
        reason=f"Polymarket={score:.0f} ({len(weighted)} mkts; {short_q})",
    )


def intersection_note(
    news_score: float,
    polymarket: PolymarketAssessment | None,
    *,
    reddit_score: float | None = None,
    reddit_mentions: int = 0,
) -> str:
    """Short tag when Polymarket agrees/diverges with news (± Reddit)."""
    if polymarket is None or not polymarket.used:
        return ""
    bits: list[str] = []
    pm = polymarket.score
    if news_score >= 60 and pm >= 60:
        bits.append("intersect: news+Polymarket bullish")
    elif news_score <= 40 and pm <= 40:
        bits.append("intersect: news+Polymarket bearish")
    elif (news_score >= 60 and pm <= 40) or (news_score <= 40 and pm >= 60):
        bits.append("diverge: news vs Polymarket")
    if reddit_mentions > 0 and reddit_score is not None:
        if reddit_score >= 60 and pm >= 60:
            bits.append("intersect: reddit+Polymarket")
        elif reddit_score <= 40 and pm <= 40:
            bits.append("intersect: reddit+Polymarket bearish")
        elif (reddit_score >= 60 and pm <= 40) or (reddit_score <= 40 and pm >= 60):
            bits.append("diverge: reddit vs Polymarket")
    return "; ".join(bits)
