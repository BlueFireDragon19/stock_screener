"""Optional X (Twitter) Trump chatter via xAI Grok x_search."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, timedelta

import requests

from stock_screener.data.trump_truth import TrumpPost

logger = logging.getLogger(__name__)

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                chunks.append(str(part.get("text") or ""))
    return "\n".join(chunks).strip()


def fetch_trump_x_posts(
    *,
    api_key: str | None = None,
    model: str | None = None,
    lookback_days: int = 7,
    timeout: float = 90.0,
) -> list[TrumpPost]:
    """
    Ask Grok to summarize recent Trump / White House X posts that name
    stocks, sectors, or trade policy. Returns synthetic 'posts' for matching.
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        logger.info("Trump X via Grok skipped (no XAI_API_KEY)")
        return []

    model_name = model or os.getenv("XAI_MODEL", "grok-4-1-fast-non-reasoning")
    to_d = date.today()
    from_d = to_d - timedelta(days=lookback_days)
    prompt = (
        "Use X search for recent posts by Donald Trump (@realDonaldTrump) or "
        "about Trump tariffs/trade/Fed/energy/China that mention specific companies, "
        "tickers, or industries. "
        f"Prefer posts from {from_d.isoformat()} to {to_d.isoformat()}. "
        "Return ONLY JSON: {\"items\":[{\"text\":\"short quote or paraphrase\", "
        "\"url\":\"optional x.com url\", \"tickers\":[\"AAPL\"]}] } "
        "Max 12 items. If nothing relevant, {\"items\":[]}."
    )
    try:
        resp = requests.post(
            XAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "input": prompt,
                "tools": [{"type": "x_search"}],
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning("Trump Grok X HTTP %s: %s", resp.status_code, resp.text[:200])
            return []
        payload = resp.json()
        text = _response_text(payload)
        data = _extract_json(text) or {}
        items = data.get("items") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trump Grok X failed: %s", exc)
        return []

    posts: list[TrumpPost] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or "").strip()
        tickers = item.get("tickers") or []
        if tickers:
            body = body + " " + " ".join(f"${t}" for t in tickers if isinstance(t, str))
        if not body:
            continue
        posts.append(
            TrumpPost(
                source="x",
                post_id=f"grok-x-{i}-{hash(body) % 10_000_000}",
                published=None,
                text=body,
                url=str(item.get("url") or ""),
            )
        )
    logger.info("Trump X (Grok): %d items", len(posts))
    return posts
