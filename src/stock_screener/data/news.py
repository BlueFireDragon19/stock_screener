from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

POSITIVE = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "record",
    "upgrade",
    "upgraded",
    "buyback",
    "raises",
    "raised",
    "growth",
    "profit",
    "strong",
    "outperform",
    "breakthrough",
    "approval",
    "approved",
    "partnership",
    "expansion",
}

NEGATIVE = {
    "miss",
    "misses",
    "cut",
    "cuts",
    "downgrade",
    "downgraded",
    "probe",
    "fraud",
    "lawsuit",
    "sues",
    "recall",
    "bankruptcy",
    "default",
    "delist",
    "delisting",
    "investigation",
    "sec charges",
    "layoffs",
    "weak",
    "plunge",
    "plunges",
    "crash",
    "warning",
    "guidance cut",
    "going concern",
}

VETO_PHRASES = {
    "fraud",
    "bankruptcy",
    "chapter 11",
    "delisting",
    "going concern",
    "sec charges",
    "accounting scandal",
}


@dataclass
class NewsAssessment:
    score: float  # 0–100
    veto: bool
    headline_count: int
    top_headline: str
    reason: str


def _parse_news_time(item: dict) -> datetime | None:
    ts = item.get("providerPublishTime") or item.get("pubDate")
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _headline_text(item: dict) -> str:
    title = str(item.get("title") or "")
    summary = str(item.get("summary") or item.get("content") or "")
    return f"{title} {summary}".lower()


def score_headlines(items: list[dict], news_hours: int) -> NewsAssessment:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=news_hours)
    recent: list[tuple[datetime | None, str]] = []
    for item in items:
        published = _parse_news_time(item)
        if published is not None and published < cutoff:
            continue
        text = _headline_text(item)
        if not text.strip():
            continue
        recent.append((published, text))

    if not recent:
        return NewsAssessment(
            score=50.0,
            veto=False,
            headline_count=0,
            top_headline="",
            reason="No recent headlines (neutral)",
        )

    veto = False
    pos = neg = 0
    for _, text in recent:
        if any(p in text for p in VETO_PHRASES):
            veto = True
        # phrase-aware first
        if "guidance cut" in text or "sec charges" in text:
            neg += 2
        tokens = set(text.replace("-", " ").split())
        pos += len(tokens & POSITIVE)
        neg += len(tokens & NEGATIVE)

    raw = pos - neg
    # Map roughly from [-5, +5] → [0, 100]
    score = 50.0 + max(-5.0, min(5.0, float(raw))) * 10.0
    top = recent[0][1][:120]
    if veto:
        reason = "News veto triggered"
    elif score >= 60:
        reason = "Leaning positive headlines"
    elif score <= 40:
        reason = "Leaning negative headlines"
    else:
        reason = "Mixed / neutral headlines"

    return NewsAssessment(
        score=score,
        veto=veto,
        headline_count=len(recent),
        top_headline=top,
        reason=reason,
    )


def fetch_news_assessment(
    ticker: str,
    news_hours: int = 72,
    pause: float = 0.15,
) -> NewsAssessment:
    """Pull recent Yahoo Finance news for a ticker (free, no API key)."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("News fetch failed for %s: %s", ticker, exc)
        items = []
    if pause:
        time.sleep(pause)
    return score_headlines(items, news_hours=news_hours)
