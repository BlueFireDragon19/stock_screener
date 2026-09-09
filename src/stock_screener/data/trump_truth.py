"""Truth Social posts via trumpstruth.org public RSS mirror."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

FEED_URL = "https://trumpstruth.org/feed"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener/0.1"
)
_TAG = re.compile(r"<[^>]+>")


@dataclass
class TrumpPost:
    source: str  # truth_social | x | news
    post_id: str
    published: datetime | None
    text: str
    url: str


def _strip_html(raw: str) -> str:
    text = unescape(_TAG.sub(" ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def fetch_truth_posts(
    *,
    max_items: int = 80,
    timeout: float = 20.0,
) -> list[TrumpPost]:
    """Pull recent @realDonaldTrump posts from trumpstruth.org RSS."""
    try:
        resp = requests.get(
            FEED_URL,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml,application/xml"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning("Truth RSS HTTP %s", resp.status_code)
            return []
        root = ET.fromstring(resp.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Truth RSS fetch failed: %s", exc)
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    posts: list[TrumpPost] = []
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        desc = item.findtext("description") or ""
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or link or title)[:120]
        pub_raw = item.findtext("pubDate")
        published = None
        if pub_raw:
            try:
                published = parsedate_to_datetime(pub_raw)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                published = None
        text = _strip_html(f"{title}. {desc}" if title else desc)
        if not text:
            continue
        posts.append(
            TrumpPost(
                source="truth_social",
                post_id=guid,
                published=published,
                text=text,
                url=link,
            )
        )
    logger.info("Truth Social mirror: %d posts", len(posts))
    return posts
