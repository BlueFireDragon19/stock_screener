"""White House official RSS (policy text; free X alternative)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from xml.etree import ElementTree as ET

import requests

from stock_screener.data.trump_truth import TrumpPost

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener/0.1"
)
_TAG = re.compile(r"<[^>]+>")

# Official WH feeds that often move markets (tariffs, EO, energy, trade)
FEEDS = (
    "https://www.whitehouse.gov/news/feed/",
    "https://www.whitehouse.gov/presidential-actions/feed/",
    "https://www.whitehouse.gov/briefings-statements/feed/",
)

# Keep items that look market-relevant (WH feed is noisy)
_MARKET_HINTS = re.compile(
    r"\b("
    r"tariff|trade|china|economy|jobs|energy|oil|gas|drill|bank|fed|"
    r"interest\s+rate|crypto|bitcoin|semiconductor|chip|steel|aluminum|"
    r"auto|manufactur|immigration|border|pharma|drug|defense|military|"
    r"executive\s+order|sanction|export|import|market|stock|invest|"
    r"equity|stake|chips\s+act|micron|dell|intel|foundry"
    r")\b",
    re.IGNORECASE,
)


def _strip_html(raw: str) -> str:
    text = unescape(_TAG.sub(" ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def _parse_feed(url: str, *, max_items: int, timeout: float) -> list[TrumpPost]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml,application/xml"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("WH RSS HTTP %s for %s", resp.status_code, url)
            return []
        root = ET.fromstring(resp.content)
    except Exception as exc:  # noqa: BLE001
        logger.debug("WH RSS failed (%s): %s", url, exc)
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    out: list[TrumpPost] = []
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = item.findtext("description") or ""
        # content:encoded often has the full body
        content = ""
        for child in list(item):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"encoded", "content"} and child.text:
                content = child.text
                break
        guid = (item.findtext("guid") or link or title)[:160]
        pub_raw = item.findtext("pubDate")
        published = None
        if pub_raw:
            try:
                published = parsedate_to_datetime(pub_raw)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                published = None
        text = _strip_html(f"{title}. {desc} {content}")
        if not text:
            continue
        # Prefer market-relevant items; always keep presidential actions titles short-pass
        if "presidential-actions" not in url and not _MARKET_HINTS.search(text):
            continue
        out.append(
            TrumpPost(
                source="white_house",
                post_id=guid,
                published=published,
                text=text[:2500],
                url=link,
            )
        )
    return out


def fetch_white_house(*, max_per_feed: int = 40, timeout: float = 20.0) -> list[TrumpPost]:
    """Pull recent White House news / EO / briefings relevant to markets."""
    seen: set[str] = set()
    posts: list[TrumpPost] = []
    for feed in FEEDS:
        for p in _parse_feed(feed, max_items=max_per_feed, timeout=timeout):
            key = p.text[:140].lower()
            if key in seen:
                continue
            seen.add(key)
            posts.append(p)
    logger.info("White House RSS: %d items", len(posts))
    return posts
