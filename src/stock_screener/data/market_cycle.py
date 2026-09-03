"""Market-cycle overlays: CAPE-ish and Buffett Indicator (best-effort free data)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener/0.1"
)


@dataclass(frozen=True)
class MarketCycle:
    cape: float | None
    buffett_pct: float | None  # market / GDP * 100
    label: str  # cheap | fair | expensive
    note: str
    # How value mode should behave
    min_value_score: float
    top_n_scale: float  # multiply top_n (e.g. 0.7 when expensive)


def _fetch_multpl_cape(timeout: float = 12.0) -> float | None:
    """Shiller CAPE from multpl.com (current reading)."""
    try:
        resp = requests.get(
            "https://www.multpl.com/shiller-pe",
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        # Current value usually in first big number on page, e.g. "37.54"
        m = re.search(
            r'class="info"[^>]*>\s*([0-9]+(?:\.[0-9]+)?)\s*<',
            resp.text,
            re.I,
        )
        if not m:
            m = re.search(r"Shiller PE[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)", resp.text, re.I)
        if not m:
            return None
        return float(m.group(1))
    except Exception as exc:  # noqa: BLE001
        logger.debug("CAPE fetch failed: %s", exc)
        return None


def _fetch_buffett_pct(timeout: float = 12.0) -> float | None:
    """
    Approx Buffett Indicator = Wilshire 5000 / US GDP.
    Uses Yahoo ^W5000 (or VTI* proxy) and FRED GDP via stooq/yahoo if needed.
    Best-effort: Wilshire level vs latest annual US GDP from a public page.
    """
    try:
        # Wilshire 5000 Full Cap (Yahoo)
        w = yf.Ticker("^W5000").history(period="5d", auto_adjust=True)
        if w.empty:
            w = yf.Ticker("VTI").history(period="5d", auto_adjust=True)
            if w.empty:
                return None
            # VTI is an ETF — not total market; skip rather than mislead
            return None
        wilshire = float(w["Close"].iloc[-1])
        # Wilshire index ≈ total market cap in billions when scaled historically;
        # CurrentMarketValuation / GuruFocus often quote ratio directly.
        # Fallback scrape of a simple public summary if available.
        resp = requests.get(
            "https://www.gurufocus.com/economic_indicators/3002/buffett-indicator-market-cap-to-gdp",
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code == 200:
            m = re.search(
                r"(?:Buffett Indicator|Market Cap to GDP)[^0-9%]{0,80}([0-9]+(?:\.[0-9]+)?)\s*%",
                resp.text,
                re.I,
            )
            if m:
                return float(m.group(1))
        # Last-resort rough: Wilshire points often near market-cap-$B; use ~28T GDP
        # only if wilshire looks like absolute level > 10_000
        if wilshire > 10_000:
            gdp_b = 28_000.0  # ~US nominal GDP $B ballpark; static fallback
            return (wilshire / gdp_b) * 100.0
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Buffett indicator fetch failed: %s", exc)
        return None


def assess_market_cycle() -> MarketCycle:
    """
    Classify equity market valuation for value-mode risk dial.

    CAPE: <20 cheap, 20–28 fair, >28 expensive (rough bands).
    Buffett %: <100 cheap, 100–140 fair, >140 expensive.
    """
    cape = _fetch_multpl_cape()
    buffett = _fetch_buffett_pct()

    votes_expensive = 0
    votes_cheap = 0
    bits: list[str] = []

    if cape is not None:
        bits.append(f"CAPE={cape:.1f}")
        if cape >= 28:
            votes_expensive += 1
        elif cape <= 20:
            votes_cheap += 1
    else:
        bits.append("CAPE=n/a")

    if buffett is not None:
        bits.append(f"Buffett={buffett:.0f}%")
        if buffett >= 140:
            votes_expensive += 1
        elif buffett <= 100:
            votes_cheap += 1
    else:
        bits.append("Buffett=n/a")

    if votes_expensive > votes_cheap:
        label = "expensive"
        min_score, scale = 62.0, 0.7
    elif votes_cheap > votes_expensive:
        label = "cheap"
        min_score, scale = 50.0, 1.0
    else:
        label = "fair"
        min_score, scale = 55.0, 0.85

    return MarketCycle(
        cape=cape,
        buffett_pct=buffett,
        label=label,
        note="; ".join(bits) + f" → {label}",
        min_value_score=min_score,
        top_n_scale=scale,
    )
