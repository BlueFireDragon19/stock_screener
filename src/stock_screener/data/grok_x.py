from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta

import requests

logger = logging.getLogger(__name__)

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"


@dataclass
class GrokXAssessment:
    score: float  # 0–100
    label: str
    summary: str
    used: bool


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
    # Responses API shapes vary; collect output_text-like fields.
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in {
                "output_text",
                "text",
            }:
                chunks.append(str(part.get("text") or ""))
    return "\n".join(chunks).strip()


def fetch_grok_x_assessment(
    ticker: str,
    api_key: str | None = None,
    model: str | None = None,
    lookback_days: int = 7,
    timeout: float = 90.0,
) -> GrokXAssessment:
    """
    Optional X sentiment via xAI Grok `x_search` (requires XAI_API_KEY).

    Intended for top survivors only — not full-universe scans.
    """
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        return GrokXAssessment(50.0, "skipped", "XAI_API_KEY not set", used=False)

    model_name = model or os.getenv("XAI_MODEL", "grok-4-1-fast-non-reasoning")
    to_d = date.today()
    from_d = to_d - timedelta(days=lookback_days)
    prompt = (
        f"Use X search for recent investor chatter about ${ticker} stock. "
        f"Return ONLY compact JSON with keys score (0-100, higher=more bullish), "
        f"label (bullish|neutral|bearish), summary (one short sentence)."
    )
    body = {
        "model": model_name,
        "input": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "x_search",
                "from_date": from_d.isoformat(),
                "to_date": to_d.isoformat(),
            }
        ],
    }
    try:
        resp = requests.post(
            XAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            logger.warning(
                "Grok x_search failed for %s: HTTP %s %s",
                ticker,
                resp.status_code,
                resp.text[:200],
            )
            return GrokXAssessment(
                50.0, "error", f"HTTP {resp.status_code}", used=True
            )
        text = _response_text(resp.json())
        data = _extract_json(text) or {}
        score = float(data.get("score", 50))
        label = str(data.get("label", "neutral"))
        summary = str(data.get("summary", text[:160] or "No summary"))
        score = max(0.0, min(100.0, score))
        return GrokXAssessment(score, label, summary, used=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Grok x_search error for %s: %s", ticker, exc)
        return GrokXAssessment(50.0, "error", str(exc)[:120], used=True)


def enrich_top_with_grok(
    tickers: list[str],
    api_key: str | None = None,
) -> dict[str, GrokXAssessment]:
    out: dict[str, GrokXAssessment] = {}
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        logger.info("Skipping Grok X enrichment (no XAI_API_KEY)")
        return out
    logger.info("Grok X enrichment for %d tickers", len(tickers))
    for t in tickers:
        out[t] = fetch_grok_x_assessment(t, api_key=key)
    return out
