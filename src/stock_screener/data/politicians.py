from __future__ import annotations

import logging
import math
from collections import defaultdict

from stock_screener.data.politicians_capitol import fetch_capitol_trades
from stock_screener.data.politicians_house import fetch_house_trades
from stock_screener.data.politicians_models import (
    PoliticianTickerScore,
    PoliticianTrade,
)
from stock_screener.data.politicians_senate import fetch_senate_trades

logger = logging.getLogger(__name__)


def freshness_weight(filed_after_days: int | None, lag_cap: int = 45) -> float:
    """
    Shorter filing lag → higher weight.
    0 days → 1.0, 45 days → 0.0, missing → 0.35 (mild penalty).
    """
    if filed_after_days is None:
        return 0.35
    if filed_after_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (filed_after_days / float(lag_cap)))


def trade_signed_weight(trade: PoliticianTrade, lag_cap: int = 45) -> float:
    if trade.tx_type not in {"buy", "sell"}:
        return 0.0
    sign = 1.0 if trade.tx_type == "buy" else -1.0
    size_w = math.log1p(max(trade.size_mid, 1.0))
    fresh = freshness_weight(trade.filed_after_days, lag_cap=lag_cap)
    # 25% base + 75% freshness so late filings barely move the needle
    return sign * size_w * (0.25 + 0.75 * fresh)


def _dedupe(trades: list[PoliticianTrade]) -> list[PoliticianTrade]:
    seen: set[tuple] = set()
    out: list[PoliticianTrade] = []
    for t in trades:
        key = (
            t.ticker,
            t.politician.lower(),
            t.tx_type,
            t.trade_date.isoformat() if t.trade_date else "",
            round(t.size_mid, -2),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def collect_politician_trades(
    *,
    capitol_pages: int = 5,
    house_max_filings: int = 35,
    enable_capitol: bool = True,
    enable_house: bool = True,
    enable_senate: bool = True,
) -> list[PoliticianTrade]:
    trades: list[PoliticianTrade] = []
    if enable_capitol:
        trades.extend(fetch_capitol_trades(pages=capitol_pages))
    if enable_house:
        trades.extend(fetch_house_trades(max_filings=house_max_filings))
    if enable_senate:
        trades.extend(fetch_senate_trades())
    deduped = _dedupe(trades)
    logger.info(
        "Politicians: %d raw → %d deduped trades (sources=%s)",
        len(trades),
        len(deduped),
        sorted({t.source for t in deduped}),
    )
    return deduped


def score_politician_tickers(
    trades: list[PoliticianTrade],
    *,
    lag_cap: int = 45,
    min_trades: int = 1,
) -> dict[str, PoliticianTickerScore]:
    by_ticker: dict[str, list[PoliticianTrade]] = defaultdict(list)
    for t in trades:
        if t.ticker:
            by_ticker[t.ticker].append(t)

    out: dict[str, PoliticianTickerScore] = {}
    raw_scores: dict[str, float] = {}

    for ticker, items in by_ticker.items():
        buy_w = sell_w = 0.0
        buys = sells = 0
        lags: list[int] = []
        pols: set[str] = set()
        sources: set[str] = set()
        for tr in items:
            w = trade_signed_weight(tr, lag_cap=lag_cap)
            if tr.tx_type == "buy":
                buy_w += abs(w)
                buys += 1
            elif tr.tx_type == "sell":
                sell_w += abs(w)
                sells += 1
            if tr.filed_after_days is not None:
                lags.append(tr.filed_after_days)
            if tr.politician:
                pols.add(tr.politician)
            sources.add(tr.source)
        net = buy_w - sell_w
        raw_scores[ticker] = net
        avg_lag = sum(lags) / len(lags) if lags else None
        ok = len(items) >= min_trades
        why_bits = [
            f"buys={buys}",
            f"sells={sells}",
            f"net_w={net:.2f}",
        ]
        if avg_lag is not None:
            why_bits.append(f"avg_filed_after={avg_lag:.0f}d")
        if pols:
            why_bits.append("pols=" + ",".join(sorted(pols)[:4]))
        out[ticker] = PoliticianTickerScore(
            ticker=ticker,
            score=50.0,  # filled after normalize
            buy_weight=round(buy_w, 3),
            sell_weight=round(sell_w, 3),
            trade_count=len(items),
            buy_count=buys,
            sell_count=sells,
            avg_filed_after=round(avg_lag, 1) if avg_lag is not None else None,
            politicians=", ".join(sorted(pols)[:8]),
            sources=",".join(sorted(sources)),
            why="; ".join(why_bits),
            pass_filters=ok,
            filter_reason="" if ok else "no trades",
        )

    if not raw_scores:
        return out

    # Map net weights to 0–100 around neutral 50
    vals = list(raw_scores.values())
    max_abs = max(abs(v) for v in vals) or 1.0
    for ticker, net in raw_scores.items():
        # net/max_abs in [-1,1] → score in [0,100]
        score = 50.0 + 50.0 * (net / max_abs)
        snap = out[ticker]
        snap.score = round(max(0.0, min(100.0, score)), 2)
        # Prefer names with buy-heavy fresh flow
        if snap.pass_filters and snap.buy_count == 0 and snap.sell_count > 0:
            # still pass but low score already
            pass
    return out
