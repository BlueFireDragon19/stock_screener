"""Google News RSS headlines that quote X/Twitter for a ticker (free X proxy)."""

from __future__ import annotations

import logging
import re
import time
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests

from stock_screener.data.news import NewsAssessment, score_headlines

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener/0.1"
)
_TAG = re.compile(r"<[^>]+>")


def _strip_html(raw: str) -> str:
    text = unescape(_TAG.sub(" ", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def fetch_x_quote_news_assessment(
    ticker: str,
    *,
    news_hours: int = 72,
    pause: float = 0.05,
    timeout: float = 15.0,
) -> NewsAssessment:
    """Score recent Google News headlines that mention ticker + X/Twitter/tweet."""
    sym = ticker.upper().strip()
    q = f'("{sym}" OR ${sym}) (Twitter OR "on X" OR tweet OR tweets)'
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    )
    items: list[dict] = []
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("X-quote news HTTP %s for %s", resp.status_code, sym)
        else:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item")[:20]:
                    title = _strip_html(item.findtext("title") or "")
                    desc = _strip_html(item.findtext("description") or "")
                    if not title:
                        continue
                    pub_raw = item.findtext("pubDate")
                    pub_ts = None
                    if pub_raw:
                        try:
                            dt = parsedate_to_datetime(pub_raw)
                            pub_ts = dt.timestamp()
                        except Exception:  # noqa: BLE001
                            pub_ts = None
                    items.append(
                        {
                            "title": title,
                            "summary": desc,
                            "providerPublishTime": pub_ts,
                        }
                    )
    except Exception as exc:  # noqa: BLE001
        logger.debug("X-quote news failed for %s: %s", sym, exc)
        items = []
    if pause:
        time.sleep(pause)
    assessment = score_headlines(items, news_hours=news_hours)
    if assessment.headline_count == 0:
        return NewsAssessment(
            score=50.0,
            veto=False,
            headline_count=0,
            top_headline="",
            reason="No X-quoted headlines (neutral)",
        )
    return NewsAssessment(
        score=assessment.score,
        veto=assessment.veto,
        headline_count=assessment.headline_count,
        top_headline=assessment.top_headline,
        reason=f"X-quoted: {assessment.reason}",
    )


def blend_yahoo_and_x_quote(
    yahoo: NewsAssessment,
    x_quote: NewsAssessment | None,
    *,
    x_weight: float = 0.35,
) -> NewsAssessment:
    """Keep Yahoo primary; tilt with X-quoting RSS when it has headlines."""
    if x_quote is None or x_quote.headline_count <= 0:
        return yahoo
    if yahoo.headline_count <= 0:
        return NewsAssessment(
            score=x_quote.score,
            veto=x_quote.veto,
            headline_count=x_quote.headline_count,
            top_headline=x_quote.top_headline,
            reason=x_quote.reason,
        )
    w = max(0.0, min(0.5, float(x_weight)))
    score = (1.0 - w) * yahoo.score + w * x_quote.score
    veto = yahoo.veto or x_quote.veto
    top = yahoo.top_headline or x_quote.top_headline
    reason = (
        f"{yahoo.reason}; x-quote={x_quote.score:.0f} "
        f"({x_quote.headline_count}h)"
    )
    if veto:
        reason = "News veto triggered (Yahoo and/or X-quote)"
    return NewsAssessment(
        score=round(float(score), 2),
        veto=veto,
        headline_count=yahoo.headline_count + x_quote.headline_count,
        top_headline=top,
        reason=reason,
    )
