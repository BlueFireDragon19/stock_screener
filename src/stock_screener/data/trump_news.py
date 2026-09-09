"""Trump / policy market headlines via Google News RSS (free)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests

from stock_screener.data.trump_truth import TrumpPost

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener/0.1"
)
_TAG = re.compile(r"<[^>]+>")

QUERIES = (
    "Trump stock market OR tariff OR Fed OR oil OR China",
    "Trump executive order OR regulation OR industry",
    "White House Trump economy OR trade",
    # Catch X/Twitter posts as quoted by news outlets (free X proxy)
    'Trump (tariff OR Fed OR China OR oil) (X OR Twitter OR "on X" OR tweet)',
    "Trump posts on X OR Truth Social tariff OR trade OR stocks",
    "Trump bitcoin OR crypto OR semiconductor OR steel OR aluminum",
    # Govt equity / industrial-policy stakes in private companies
    'Trump OR "White House" OR Commerce ("equity stake" OR "government stake" OR "CHIPS Act" OR "minority stake")',
    "government equity stake Intel OR Micron OR Dell OR semiconductor",
    'Trump (Micron OR Dell OR Intel) (investment OR stake OR CHIPS OR funding)',
    '"government investment" OR "federal equity" OR "Uncle Sam" shareholder Trump',
)


def _strip_html(raw: str) -> str:
    text = unescape(_TAG.sub(" ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def _fetch_query(query: str, timeout: float = 15.0) -> list[TrumpPost]:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("Google News RSS HTTP %s for %s", resp.status_code, query)
            return []
        root = ET.fromstring(resp.content)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Google News RSS failed (%s): %s", query, exc)
        return []

    channel = root.find("channel")
    if channel is None:
        return []
    out: list[TrumpPost] = []
    for item in channel.findall("item")[:25]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = item.findtext("description") or ""
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
        text = _strip_html(f"{title}. {desc}")
        if not text:
            continue
        out.append(
            TrumpPost(
                source="news",
                post_id=guid,
                published=published,
                text=text,
                url=link,
            )
        )
    return out


def fetch_trump_news(*, timeout: float = 15.0) -> list[TrumpPost]:
    """Aggregate Trump/policy market headlines from Google News RSS."""
    seen: set[str] = set()
    posts: list[TrumpPost] = []
    for q in QUERIES:
        for p in _fetch_query(q, timeout=timeout):
            key = p.text[:120].lower()
            if key in seen:
                continue
            seen.add(key)
            posts.append(p)
    logger.info("Trump news RSS: %d headlines", len(posts))
    return posts
