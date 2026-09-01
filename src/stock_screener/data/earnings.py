from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class EarningsAssessment:
    days_to_earnings: int | None
    score: float  # 0–100 support-mode score
    reason: str
    catalyst_score: float = 50.0
    catalyst_reason: str = "Earnings date unknown"


def _nearest_earnings_days(ticker: str) -> int | None:
    """Signed days to nearest earnings: negative = already reported."""
    t = yf.Ticker(ticker)
    now = datetime.now(timezone.utc)
    candidates: list[datetime] = []

    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            idx = pd.to_datetime(ed.index, utc=True)
            candidates.extend(ts.to_pydatetime() for ts in idx)
    except Exception:  # noqa: BLE001
        pass

    try:
        cal = t.calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(raw, list):
                for item in raw:
                    candidates.append(pd.to_datetime(item, utc=True).to_pydatetime())
            elif raw is not None:
                candidates.append(pd.to_datetime(raw, utc=True).to_pydatetime())
        elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"].iloc[0]
            candidates.append(pd.to_datetime(val, utc=True).to_pydatetime())
    except Exception:  # noqa: BLE001
        pass

    if not candidates:
        return None

    # Prefer next future date; else most recent past
    future = [d for d in candidates if d >= now]
    if future:
        nxt = min(future)
        return int((nxt - now).total_seconds() // 86400)
    past = max(d for d in candidates if d < now)
    return int((past - now).total_seconds() // 86400)


def score_earnings(days: int | None) -> EarningsAssessment:
    """Support-mode: avoid binary event risk near the print."""
    if days is None:
        return EarningsAssessment(
            None, 50.0, "Earnings date unknown", 50.0, "Earnings date unknown"
        )
    if days < 0:
        ago = -days
        support = EarningsAssessment(
            days, 55.0, f"Reported {ago}d ago", 50.0, ""
        )
    elif days <= 2:
        support = EarningsAssessment(
            days, 25.0, f"Earnings in {days}d (high risk)", 50.0, ""
        )
    elif days <= 7:
        support = EarningsAssessment(days, 45.0, f"Earnings in {days}d", 50.0, "")
    elif days <= 21:
        support = EarningsAssessment(
            days, 70.0, f"Earnings in {days}d (setup window)", 50.0, ""
        )
    else:
        support = EarningsAssessment(days, 55.0, f"Earnings in {days}d", 50.0, "")

    # Catalyst-mode: reward fresh post-print or near-term attention window
    if days < 0:
        ago = -days
        if ago <= 5:
            cat_score, cat_reason = 85.0, f"Post-earnings impulse window ({ago}d ago)"
        elif ago <= 14:
            cat_score, cat_reason = 65.0, f"Recent earnings ({ago}d ago)"
        else:
            cat_score, cat_reason = 45.0, f"Earnings {ago}d ago"
    elif days <= 2:
        cat_score, cat_reason = 55.0, f"Earnings in {days}d (event risk)"
    elif days <= 10:
        cat_score, cat_reason = 75.0, f"Pre-earnings attention ({days}d)"
    elif days <= 21:
        cat_score, cat_reason = 60.0, f"Earnings in {days}d"
    else:
        cat_score, cat_reason = 40.0, f"Earnings far ({days}d)"

    return EarningsAssessment(
        days,
        support.score,
        support.reason,
        cat_score,
        cat_reason,
    )


def fetch_earnings_assessment(ticker: str) -> EarningsAssessment:
    days = _nearest_earnings_days(ticker)
    return score_earnings(days)


def fetch_earnings_map(
    tickers: list[str],
    max_workers: int = 6,
) -> dict[str, EarningsAssessment]:
    out: dict[str, EarningsAssessment] = {}
    if not tickers:
        return out
    logger.info("Fetching earnings dates for %d tickers via yfinance", len(tickers))

    def _one(t: str) -> tuple[str, EarningsAssessment]:
        return t, fetch_earnings_assessment(t)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, assessment = fut.result()
            out[t] = assessment
    return out
