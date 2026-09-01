from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# Browser-like UA; Reddit blocks many default python-requests agents with 403.
USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener/0.1",
)

DEFAULT_SUBREDDITS = (
    "stocks",
    "investing",
    "StockMarket",
    "wallstreetbets",
    "EarningsWhispers",
)

POSITIVE = {
    "moon",
    "calls",
    "bullish",
    "breakout",
    "undervalued",
    "buy",
    "long",
    "beat",
    "upgrade",
    "rally",
}
NEGATIVE = {
    "puts",
    "bearish",
    "crash",
    "sell",
    "short",
    "overvalued",
    "baghold",
    "dilution",
    "fraud",
    "bankruptcy",
}

_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
_WORD = re.compile(r"\b([A-Z]{2,5})\b")


@dataclass
class RedditAssessment:
    score: float  # 0–100
    mentions: int
    reason: str


def _oauth_token(timeout: float = 15.0) -> str | None:
    """Optional free Reddit script/app credentials."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning("Reddit OAuth failed: HTTP %s", resp.status_code)
            return None
        return str(resp.json().get("access_token") or "") or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reddit OAuth error: %s", exc)
        return None


def _listing_urls(subreddit: str, limit: int = 50) -> list[str]:
    # Prefer old.reddit.com — often less aggressive bot blocking than www
    hosts = (
        f"https://old.reddit.com/r/{subreddit}",
        f"https://www.reddit.com/r/{subreddit}",
    )
    urls: list[str] = []
    for base in hosts:
        urls.extend(
            [
                f"{base}/hot.json?limit={limit}&raw_json=1",
                f"{base}/new.json?limit={limit}&raw_json=1",
            ]
        )
    return urls


def _fetch_posts(
    url: str,
    token: str | None,
    timeout: float = 15.0,
) -> list[dict]:
    headers = {"User-Agent": USER_AGENT}
    fetch_url = url
    if token:
        headers["Authorization"] = f"bearer {token}"
        fetch_url = url.replace("https://old.reddit.com", "https://oauth.reddit.com")
        fetch_url = fetch_url.replace(
            "https://www.reddit.com", "https://oauth.reddit.com"
        )
    try:
        resp = requests.get(fetch_url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        logger.debug("Reddit request failed %s: %s", fetch_url, exc)
        return []
    if resp.status_code != 200:
        logger.debug("Reddit %s → HTTP %s", fetch_url, resp.status_code)
        return []
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "json" not in ctype and not resp.text.lstrip().startswith("{"):
        logger.debug("Reddit %s → non-JSON body", fetch_url)
        return []
    try:
        data = resp.json()
    except ValueError:
        logger.debug("Reddit %s → JSON parse error", fetch_url)
        return []
    children = data.get("data", {}).get("children", [])
    return [c.get("data", {}) for c in children if isinstance(c, dict)]


def build_reddit_mention_index(
    known_tickers: set[str],
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
    pause: float = 0.5,
) -> dict[str, list[str]]:
    """
    Free Reddit scan via public JSON (or OAuth if REDDIT_CLIENT_ID/SECRET set).
    """
    mentions: dict[str, list[str]] = defaultdict(list)
    block = {
        "A",
        "I",
        "AM",
        "PM",
        "CEO",
        "CFO",
        "IPO",
        "ETF",
        "USD",
        "GDP",
        "AI",
        "EV",
        "DD",
        "YOLO",
        "EPS",
        "ATH",
        "ATL",
        "IMO",
        "TBH",
        "FOMO",
        "USA",
        "FOR",
        "THE",
        "AND",
        "ARE",
        "YOU",
        "ALL",
        "NEW",
        "TOP",
        "BIG",
        "NOW",
        "OUT",
        "HAS",
        "WAS",
        "BUY",
        "SELL",
        "HOLD",
        "CALL",
        "PUT",
        "IT",
        "OR",
        "ON",
        "TO",
        "IN",
        "OF",
        "IS",
        "BE",
        "AT",
        "BY",
        "VS",
        "RH",
        "WSB",
    }

    token = _oauth_token()
    if token:
        logger.info("Reddit using OAuth client credentials")
    else:
        logger.info("Reddit using public JSON (set REDDIT_CLIENT_ID/SECRET if 403s)")

    for sub in subreddits:
        got_any = False
        for url in _listing_urls(sub):
            posts = _fetch_posts(url, token=token)
            time.sleep(pause)
            if not posts:
                continue
            got_any = True
            for post in posts:
                title = str(post.get("title") or "")
                text = f"{title} {post.get('selftext') or ''}"
                found: set[str] = set()
                for m in _CASHTAG.findall(title.upper()):
                    if m in known_tickers:
                        found.add(m)
                for m in _WORD.findall(title.upper()):
                    if m in block:
                        continue
                    if m in known_tickers:
                        found.add(m)
                for ticker in found:
                    mentions[ticker].append(text.lower()[:240])
            # One successful host/listing style is enough per subreddit flavor
            if got_any and "hot.json" in url:
                break
        logger.info("Reddit scanned r/%s (hits=%s)", sub, "yes" if got_any else "no")

    logger.info("Reddit mention index: %d tickers with hits", len(mentions))
    return dict(mentions)


def score_reddit_mentions(titles: list[str] | None) -> RedditAssessment:
    if not titles:
        return RedditAssessment(50.0, 0, "No Reddit mentions")

    pos = neg = 0
    for text in titles:
        tokens = set(text.replace("$", " ").split())
        pos += len(tokens & POSITIVE)
        neg += len(tokens & NEGATIVE)

    raw = pos - neg
    attention = min(15.0, len(titles) * 3.0)
    score = 50.0 + attention + max(-20.0, min(20.0, float(raw) * 5.0))
    score = max(0.0, min(100.0, score))
    if score >= 60:
        reason = f"{len(titles)} Reddit mentions (lean bullish)"
    elif score <= 40:
        reason = f"{len(titles)} Reddit mentions (lean bearish)"
    else:
        reason = f"{len(titles)} Reddit mentions (mixed)"
    return RedditAssessment(score, len(titles), reason)


def assess_tickers_from_index(
    tickers: list[str],
    index: dict[str, list[str]],
) -> dict[str, RedditAssessment]:
    return {t: score_reddit_mentions(index.get(t)) for t in tickers}
