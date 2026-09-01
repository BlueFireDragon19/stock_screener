from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return raw
    if len(tickers) == 1:
        t = tickers[0]
        frame = raw.copy()
        if not isinstance(frame.columns, pd.MultiIndex):
            frame.columns = pd.MultiIndex.from_product([frame.columns, [t]])
        return frame
    return raw


def download_prices(
    tickers: list[str],
    history_days: int = 320,
    chunk_size: int = 150,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV for many tickers in chunks (free via yfinance).

    Returns MultiIndex columns (field, ticker).
    """
    if not tickers:
        return pd.DataFrame()

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=history_days + 40)
    frames: list[pd.DataFrame] = []

    logger.info(
        "Downloading prices for %d tickers via yfinance (chunk=%d)",
        len(tickers),
        chunk_size,
    )
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
        logger.info("Price chunk %d–%d / %d", i + 1, i + len(batch), len(tickers))
        raw = yf.download(
            tickers=batch,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            group_by="column",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        part = _normalize_download(raw, batch)
        if not part.empty:
            frames.append(part)

    if not frames:
        logger.warning("No price data returned")
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0]

    # Align on index; columns are MultiIndex (field, ticker)
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, how="outer")
    return out


def series_for(prices: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    if isinstance(prices.columns, pd.MultiIndex):
        if (field, ticker) in prices.columns:
            return prices[(field, ticker)].dropna()
        if (ticker, field) in prices.columns:
            return prices[(ticker, field)].dropna()
    if field in prices.columns:
        return prices[field].dropna()
    return pd.Series(dtype=float)


def latest_vix() -> float | None:
    """Free VIX spot via Yahoo Finance (^VIX)."""
    try:
        hist = yf.Ticker("^VIX").history(period="5d", auto_adjust=True)
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("VIX download failed: %s", exc)
        return None
