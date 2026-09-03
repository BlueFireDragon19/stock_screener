#!/usr/bin/env python3
"""
YTD backtest: support vs catalyst vs enhanced value sketch.

Value mode is EXPERIMENTAL and backtest-only — does not replace live CLI modes.
Yahoo fundamentals are current snapshots (look-ahead on multiples); disclosed below.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from stock_screener.config import ScreenerConfig
from stock_screener.data.earnings import EarningsAssessment
from stock_screener.data.fundamentals import FundamentalsSnapshot, fetch_fundamentals
from stock_screener.data.market_cycle import MarketCycle, assess_market_cycle
from stock_screener.data.news import NewsAssessment
from stock_screener.data.reddit import RedditAssessment
from stock_screener.data.sentiment import MarketSentiment, RegimeParams, _regime_for
from stock_screener.data.universe import resolve_universe
from stock_screener.features import compute_technicals
from stock_screener.scoring import build_catalyst_row, build_support_row


NEUTRAL_NEWS = NewsAssessment(50.0, False, 0, "", "neutral (backtest)")
NEUTRAL_EARN = EarningsAssessment(None, 50.0, "neutral", 50.0, "neutral")
NEUTRAL_REDDIT = RedditAssessment(50.0, 0, "neutral")


# ---------------------------------------------------------------------------
# VALUE MODE SKETCH v2 (backtest-only)
#
# Hard gates:
#   - ROE ≥ 15%
#   - Cheap: valuation_score ≥ 55 OR (PE≤20 and PB<3) OR PEG ≤ 1.5
#   - Prefer FCF yield > 0 (hard require if yield available and ≤ 0 → fail)
#   - Avoid trap: D/E > 200 and current ratio < 1
#   - No chase: RSI(14) < 75
#
# Ranking (sums ≈ 1.0):
#   35% valuation multiples | 15% PEG | 15% FCF yield
#   15% profitability/margins | 10% ROE/capital | 10% balance
# Soft: +2 if RSI ≤ 50
#
# Market-cycle overlay (CAPE + Buffett Indicator):
#   expensive → higher min score, fewer names
#   cheap → lower bar, full top_n
# ---------------------------------------------------------------------------


@dataclass
class ValueScore:
    ticker: str
    score: float
    valuation: float
    peg: float | None
    fcf_yield: float | None
    roe: float | None
    pass_filters: bool
    reason: str


def _peg_component(peg: float | None) -> float:
    if peg is None or peg <= 0:
        return 50.0
    if peg < 1.0:
        return 90.0
    if peg < 1.5:
        return 75.0
    if peg < 2.0:
        return 55.0
    if peg < 3.0:
        return 40.0
    return 25.0


def _fcfy_component(fcfy: float | None) -> float:
    if fcfy is None:
        return 50.0
    if fcfy >= 0.06:
        return 90.0
    if fcfy >= 0.04:
        return 75.0
    if fcfy >= 0.02:
        return 60.0
    if fcfy > 0:
        return 45.0
    return 20.0


def score_value_mode(
    snap: FundamentalsSnapshot,
    rsi14: float | None = None,
    *,
    min_score: float = 55.0,
) -> ValueScore:
    """Dedicated value score with PEG, FCF yield, ROE≥15% quality floor."""
    val = snap.valuation_score
    peg = snap.peg
    fcfy = snap.fcf_yield
    roe = snap.roe

    if roe is None or roe < 0.15:
        return ValueScore(
            snap.ticker,
            float("nan"),
            val,
            peg,
            fcfy,
            roe,
            False,
            f"ROE<{0.15:.0%} ({roe if roe is not None else 'n/a'})",
        )

    pe = snap.trailing_pe
    fpe = snap.forward_pe
    pb = snap.price_to_book
    cheap_gate = val >= 55.0
    if not cheap_gate:
        pe_ok = any(x is not None and 0 < x <= 20 for x in (pe, fpe))
        pb_ok = pb is not None and 0 < pb < 3.0
        peg_ok = peg is not None and 0 < peg <= 1.5
        cheap_gate = (pe_ok and pb_ok) or peg_ok

    if not cheap_gate:
        return ValueScore(
            snap.ticker, float("nan"), val, peg, fcfy, roe, False, "not cheap enough"
        )

    if fcfy is not None and fcfy <= 0:
        return ValueScore(
            snap.ticker, float("nan"), val, peg, fcfy, roe, False, "FCF yield ≤ 0"
        )

    de = snap.debt_to_equity
    cr = snap.current_ratio
    if de is not None and de > 200 and (cr is None or cr < 1.0):
        return ValueScore(
            snap.ticker, float("nan"), val, peg, fcfy, roe, False, "value-trap BS"
        )

    if rsi14 is not None and rsi14 >= 75.0:
        return ValueScore(
            snap.ticker, float("nan"), val, peg, fcfy, roe, False, "RSI chase"
        )

    score = (
        0.35 * val
        + 0.15 * _peg_component(peg)
        + 0.15 * _fcfy_component(fcfy)
        + 0.15 * snap.profitability_score
        + 0.10 * snap.capital_score
        + 0.10 * snap.balance_score
    )
    if rsi14 is not None and rsi14 <= 50.0:
        score = min(100.0, score + 2.0)

    if score < min_score:
        return ValueScore(
            snap.ticker,
            float("nan"),
            val,
            peg,
            fcfy,
            roe,
            False,
            f"below cycle min {min_score:.0f}",
        )

    return ValueScore(
        snap.ticker,
        round(score, 2),
        val,
        peg,
        fcfy,
        roe,
        True,
        f"val={val:.0f}; peg={peg if peg is not None else 'n/a'}; "
        f"fcfy={f'{fcfy:.1%}' if fcfy is not None else 'n/a'}; roe={roe:.0%}",
    )


@dataclass
class BacktestResult:
    mode: str
    ytd_return_pct: float
    spy_return_pct: float
    excess_pct: float
    rebalance_count: int
    avg_holdings: float
    win_rate_pct: float
    max_drawdown_pct: float


def market_at(vix: float | None, cfg: ScreenerConfig) -> MarketSentiment:
    if vix is None:
        label, score, reason = "neutral", 50.0, "VIX unavailable"
    elif vix <= cfg.vix_risk_on:
        label = "risk-on"
        score = 80.0 + max(0.0, (cfg.vix_risk_on - vix) * 2.0)
        reason = f"VIX {vix:.1f} ≤ {cfg.vix_risk_on:.0f}"
    elif vix >= cfg.vix_risk_off:
        label = "risk-off"
        score = max(5.0, 40.0 - (vix - cfg.vix_risk_off) * 2.0)
        reason = f"VIX {vix:.1f} ≥ {cfg.vix_risk_off:.0f}"
    else:
        span = cfg.vix_risk_off - cfg.vix_risk_on
        t = (vix - cfg.vix_risk_on) / span
        score = 80.0 - t * 40.0
        label = "neutral"
        reason = f"VIX {vix:.1f} in mid range"
    regime: RegimeParams = _regime_for(label, cfg)
    return MarketSentiment(
        vix=vix,
        score=min(100.0, max(0.0, score)),
        label=label,
        reason=reason,
        regime=regime,
    )


def download_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers=tickers + ["SPY", "^VIX"],
        start=start,
        end=end,
        group_by="column",
        auto_adjust=True,
        threads=True,
        progress=False,
    )


def _col(frame: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if (field, ticker) in frame.columns:
        return frame[(field, ticker)].dropna()
    if (ticker, field) in frame.columns:
        return frame[(ticker, field)].dropna()
    return pd.Series(dtype=float)


def fetch_fund_map(tickers: list[str]) -> dict[str, FundamentalsSnapshot]:
    out: dict[str, FundamentalsSnapshot] = {}

    def _one(t: str) -> tuple[str, FundamentalsSnapshot | None]:
        return t, fetch_fundamentals(t)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_one, t) for t in tickers]
        for fut in as_completed(futs):
            t, snap = fut.result()
            if snap is not None and snap.pass_filters:
                out[t] = snap
    return out


def pick_tickers(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    universe: list[str],
    mode: str,
    cfg: ScreenerConfig,
    market: MarketSentiment,
    top_n: int,
    fund_map: dict[str, FundamentalsSnapshot] | None = None,
    cycle: MarketCycle | None = None,
) -> list[str]:
    picks: list[tuple[str, float]] = []
    min_score = cycle.min_value_score if cycle else 55.0
    for ticker in universe:
        close = _col(prices, "Close", ticker).loc[:as_of]
        high = _col(prices, "High", ticker).loc[:as_of]
        low = _col(prices, "Low", ticker).loc[:as_of]
        vol = _col(prices, "Volume", ticker).loc[:as_of]
        tech = compute_technicals(
            close,
            high,
            low,
            vol,
            donchian_window=cfg.donchian_window,
            sma_fast=cfg.sma_fast,
            sma_slow=cfg.sma_slow,
            avg_volume_window=cfg.avg_volume_window,
        )
        if tech is None:
            continue
        if tech.price < cfg.min_price or tech.avg_dollar_volume < cfg.min_avg_dollar_volume:
            continue

        if mode == "value":
            snap = (fund_map or {}).get(ticker)
            if snap is None:
                continue
            vs = score_value_mode(snap, rsi14=tech.rsi14, min_score=min_score)
            if vs.pass_filters and not np.isnan(vs.score):
                picks.append((ticker, vs.score))
            continue

        if mode == "support":
            row = build_support_row(
                ticker, tech, market, NEUTRAL_NEWS, NEUTRAL_EARN, NEUTRAL_REDDIT, None, cfg
            )
        else:
            row = build_catalyst_row(
                ticker, tech, market, NEUTRAL_NEWS, NEUTRAL_EARN, NEUTRAL_REDDIT, None, cfg
            )
        if row.pass_filters and not np.isnan(row.composite):
            picks.append((ticker, row.composite))

    picks.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in picks[:top_n]]


def run_mode_backtest(
    prices: pd.DataFrame,
    vix: pd.Series,
    universe: list[str],
    mode: str,
    cfg: ScreenerConfig,
    year_start: pd.Timestamp,
    top_n: int,
    fund_map: dict[str, FundamentalsSnapshot] | None = None,
    cycle: MarketCycle | None = None,
) -> BacktestResult:
    spy = _col(prices, "Close", "SPY").dropna()
    trading_days = spy.index[spy.index >= year_start]
    weeks = pd.Series(trading_days).groupby(trading_days.to_period("W")).max()

    effective_top = top_n
    if mode == "value" and cycle is not None:
        effective_top = max(3, int(round(top_n * cycle.top_n_scale)))

    equity = 1.0
    curve: list[tuple[pd.Timestamp, float]] = []
    period_returns: list[float] = []
    holding_counts: list[int] = []
    rebalance_dates = list(weeks)

    for i, reb_date in enumerate(rebalance_dates):
        vix_val = float(vix.loc[:reb_date].iloc[-1]) if not vix.loc[:reb_date].empty else None
        market = market_at(vix_val, cfg)
        holdings = pick_tickers(
            prices,
            reb_date,
            universe,
            mode,
            cfg,
            market,
            effective_top,
            fund_map=fund_map,
            cycle=cycle if mode == "value" else None,
        )
        holding_counts.append(len(holdings))
        if i + 1 >= len(rebalance_dates):
            break
        next_date = rebalance_dates[i + 1]
        if not holdings:
            period_returns.append(0.0)
            curve.append((next_date, equity))
            continue
        rets = []
        for t in holdings:
            s = _col(prices, "Close", t).loc[reb_date:next_date]
            if len(s) < 2:
                continue
            rets.append(float(s.iloc[-1] / s.iloc[0] - 1.0))
        port_ret = float(np.mean(rets)) if rets else 0.0
        period_returns.append(port_ret)
        equity *= 1.0 + port_ret
        curve.append((next_date, equity))

    curve_s = pd.Series({d: v for d, v in curve}).sort_index()
    spy_ytd = float(spy.loc[year_start:].iloc[-1] / spy.loc[year_start:].iloc[0] - 1.0) * 100.0
    strat_ytd = (equity - 1.0) * 100.0
    dd = 0.0
    if not curve_s.empty:
        peak = curve_s.cummax()
        dd = float(((curve_s / peak) - 1.0).min() * 100.0)
    wins = sum(1 for r in period_returns if r > 0)
    win_rate = (wins / len(period_returns) * 100.0) if period_returns else 0.0
    avg_h = float(np.mean(holding_counts)) if holding_counts else 0.0
    return BacktestResult(
        mode=mode,
        ytd_return_pct=round(strat_ytd, 2),
        spy_return_pct=round(spy_ytd, 2),
        excess_pct=round(strat_ytd - spy_ytd, 2),
        rebalance_count=len(period_returns),
        avg_holdings=round(avg_h, 1),
        win_rate_pct=round(win_rate, 1),
        max_drawdown_pct=round(dd, 2),
    )


def print_sketch(cycle: MarketCycle) -> None:
    print(
        f"""
=== VALUE MODE SKETCH v2 (backtest-only; live CLI unchanged) ===

Stock gates
  • ROE ≥ 15%
  • Cheap: val≥55 OR (PE≤20 & PB<3) OR PEG≤1.5
  • FCF yield > 0 when available
  • Skip weak BS traps; RSI < 75; soft +2 if RSI ≤ 50

Ranking
  • 35% multiples | 15% PEG | 15% FCF yield
  • 15% margins/profit | 10% capital | 10% balance

Market-cycle overlay (current)
  • {cycle.note}
  • min value score={cycle.min_value_score:.0f} · top_n scale={cycle.top_n_scale:.2f}

Caveats
  • Yahoo fundamentals CURRENT (look-ahead)
  • CAPE/Buffett are spot readings, not full historical series
  • Profit margins still snapshot (not multi-year trend yet)
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest support / catalyst / value sketch")
    parser.add_argument("--universe", default="sp500", choices=("sp500", "us"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--year", type=int, default=None, help="Calendar year start (Jan 1)")
    parser.add_argument(
        "--months",
        type=int,
        default=None,
        help="Trailing N months ending today (e.g. 12 for 1 year)",
    )
    parser.add_argument(
        "--modes",
        default="support,catalyst,value",
        help="Comma-separated: support,catalyst,value",
    )
    args = parser.parse_args()

    cycle = assess_market_cycle()
    print_sketch(cycle)

    cfg = ScreenerConfig(
        universe=args.universe,
        max_tickers=args.limit,
        require_above_sma_slow=False,
        enable_reddit=False,
        enable_grok=False,
        enable_fundamentals=False,
        enable_politicians=False,
    )
    universe = resolve_universe(limit=args.limit, universe=args.universe)

    today = datetime.now(timezone.utc).date()
    if args.months is not None:
        period_start = pd.Timestamp(today) - pd.DateOffset(months=args.months)
        period_label = f"trailing {args.months}m ({period_start.date()} → {today})"
    else:
        year = args.year or today.year
        period_start = pd.Timestamp(f"{year}-01-01")
        period_label = f"YTD {year} ({period_start.date()} → {today})"

    warmup_start = (period_start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading {len(universe)} tickers + SPY + VIX ({warmup_start} → {end})…")
    prices = download_panel(universe, warmup_start, end)
    vix = _col(prices, "Close", "^VIX")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    need_funds = "value" in modes
    fund_map: dict[str, FundamentalsSnapshot] = {}
    if need_funds:
        print(f"Fetching Yahoo fundamentals for value sketch ({len(universe)} tickers)…")
        fund_map = fetch_fund_map(universe)
        print(f"Fundamentals snapshots: {len(fund_map)}")

    print(f"\n=== BACKTEST — {period_label} ===")
    print(f"Universe: {args.universe} (limit {args.limit}) · weekly · top {args.top} EW")
    print(f"Modes: {', '.join(modes)}\n")

    results: list[BacktestResult] = []
    for mode in modes:
        res = run_mode_backtest(
            prices,
            vix,
            universe,
            mode,
            cfg,
            period_start,
            args.top,
            fund_map=fund_map if mode == "value" else None,
            cycle=cycle if mode == "value" else None,
        )
        results.append(res)

    header = (
        f"{'Mode':10} {'Return%':>8} {'SPY%':>8} {'Excess':>8} "
        f"{'WinWk%':>8} {'MaxDD%':>8} {'Avg#':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.mode:10} {r.ytd_return_pct:+8.2f} {r.spy_return_pct:+8.2f} "
            f"{r.excess_pct:+8.2f} {r.win_rate_pct:8.1f} {r.max_drawdown_pct:8.2f} "
            f"{r.avg_holdings:6.1f}"
        )

    if results:
        best = max(results, key=lambda x: x.excess_pct)
        print(f"\nBest excess vs SPY (this run): {best.mode} ({best.excess_pct:+.2f}%)")
    print("Live screener logic NOT replaced — value remains a sketch until you approve.")


if __name__ == "__main__":
    main()
