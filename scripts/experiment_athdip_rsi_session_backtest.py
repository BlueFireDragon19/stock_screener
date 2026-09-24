#!/usr/bin/env python3
"""Backtest athdip 4h RSI session variants: RTH vs extended vs calendar-4h.

Yahoo Finance facts (US equities):
  • Native ``interval=4h`` → RTH only (typically 09:30 and 13:30 ET bars).
  • ``interval=1h, prepost=False`` → RTH hourly.
  • ``interval=1h, prepost=True`` → extended (~04:00–20:00 ET), not true 24h
    overnight (Yahoo does not publish continuous overnight bars for US stocks).

Variants compared:
  rth          — Yahoo native 4h (current production athdip input)
  extended     — 1h+prepost resampled into 4h blocks during available hours
  calendar_4h  — same extended 1h data, resampled on fixed 4h ET clock edges
                 (closest proxy to “24h-style” continuous RSI)

Signal rules match live athdip: multi-year ATH setup (≤63d), min 4h RSI ≤31
over 5 calendar days, fire on the calendar day the min RSI printed.
Forward returns use daily closes (5 / 10 / 20 trading days).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_screener.config import ScreenerConfig  # noqa: E402
from stock_screener.data.prices import download_prices, series_for  # noqa: E402
from stock_screener.features import (  # noqa: E402
    compute_rsi_series,
    compute_technicals,
    min_rsi_lookback_detail,
    pct_from_high,
    recent_multi_year_ath_setup,
)
from stock_screener.data.universe import resolve_universe  # noqa: E402

ET = "America/New_York"
OUTPUT = ROOT / "output"


@dataclass
class VariantStats:
    variant: str
    n_signals: int
    n_tickers: int
    avg_rsi: float
    win_5d: float
    avg_5d: float
    med_5d: float
    win_10d: float
    avg_10d: float
    med_10d: float
    win_20d: float
    avg_20d: float
    med_20d: float
    avg_pct_ath: float


def _close_1h(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close", ticker) in raw.columns:
            s = raw[("Close", ticker)]
        elif (ticker, "Close") in raw.columns:
            s = raw[(ticker, "Close")]
        else:
            # single-ticker download sometimes drops ticker level
            s = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
    else:
        s = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
    s = s.dropna().astype(float)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s.sort_index()


def download_1h(tickers: list[str], *, period: str, prepost: bool) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    chunk = 20
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        raw = yf.download(
            tickers=batch,
            period=period,
            interval="1h",
            group_by="column",
            auto_adjust=True,
            threads=True,
            progress=False,
            prepost=prepost,
        )
        for t in batch:
            s = _close_1h(raw, t)
            if len(s) >= 30:
                out[t] = s
    return out


def download_4h_native(tickers: list[str], *, period: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    chunk = 20
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        raw = yf.download(
            tickers=batch,
            period=period,
            interval="4h",
            group_by="column",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        for t in batch:
            s = _close_1h(raw, t)  # same Close extraction
            if len(s) >= 20:
                out[t] = s
    return out


def resample_to_4h(close_1h: pd.Series, *, how: str) -> pd.Series:
    """Resample hourly closes to 4h.

    how:
      extended — all prepost hours (~04:00–20:00 ET), 4h bins
      calendar_4h — same feed; bins anchored at midnight ET (24h-style clock)
    """
    if close_1h.empty:
        return close_1h
    s = close_1h.copy()
    s.index = s.index.tz_convert(ET)
    if how == "extended":
        # Keep Yahoo extended session only (drop sparse overnight holes if any)
        hours = s.index.hour
        s = s[(hours >= 4) & (hours <= 19)]
        out = s.resample("4h", label="right", closed="right", origin="start_day").last()
    else:
        # Fixed clock bins from midnight ET (closest to continuous/24h RSI path)
        out = s.resample("4h", label="right", closed="right", origin="start_day").last()
    out = out.dropna()
    out.index = out.index.tz_convert("UTC")
    return out


def fwd_ret(daily_close: pd.Series, trigger_date, horizon: int) -> float | None:
    c = daily_close.dropna()
    if c.empty:
        return None
    # normalize index to dates
    idx_dates = [
        (ts.tz_convert(ET).date() if getattr(ts, "tzinfo", None) else pd.Timestamp(ts).date())
        for ts in c.index
    ]
    date_to_i = {d: i for i, d in enumerate(idx_dates)}
    td = trigger_date if hasattr(trigger_date, "year") else pd.Timestamp(trigger_date).date()
    # entry = close on trigger day (or next available)
    if td not in date_to_i:
        later = [d for d in idx_dates if d >= td]
        if not later:
            return None
        td = later[0]
    i0 = date_to_i[td]
    i1 = i0 + horizon
    if i1 >= len(c):
        return None
    p0 = float(c.iloc[i0])
    p1 = float(c.iloc[i1])
    if p0 <= 0:
        return None
    return (p1 / p0 - 1.0) * 100.0


def collect_signals(
    *,
    ticker: str,
    daily_high: pd.Series,
    daily_low: pd.Series,
    daily_close: pd.Series,
    daily_vol: pd.Series,
    close_4h: pd.Series,
    cfg: ScreenerConfig,
) -> list[dict]:
    if close_4h.empty or len(close_4h) < 20:
        return []
    rsi = compute_rsi_series(close_4h, period=14).dropna()
    if rsi.empty:
        return []

    def _session_date(ts) -> object:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            return t.tz_convert(ET).date()
        return t.date()

    # Only evaluate days that printed RSI ≤ max somewhere (huge speedup)
    candidate_days = sorted(
        {_session_date(ts) for ts, val in rsi.items() if float(val) <= cfg.athdip_rsi_4h_max}
    )
    rows: list[dict] = []
    seen_days: set = set()

    # Precompute date masks for daily series (tz-naive Yahoo daily)
    def _cut(s: pd.Series, d) -> pd.Series:
        if s.empty:
            return s
        if s.index.tz is not None:
            idx_dates = [i.tz_convert(ET).date() for i in s.index]
            return s.loc[[i for i, dd in zip(s.index, idx_dates) if dd <= d]]
        # normalize to date comparison without slow loop when possible
        return s.loc[: pd.Timestamp(d)]

    for d in candidate_days:
        as_of = pd.Timestamp(d)
        trig = min_rsi_lookback_detail(
            close_4h,
            as_of=as_of,
            lookback_days=cfg.athdip_rsi_lookback_days,
            period=14,
        )
        if trig is None:
            continue
        rsi_4h, trigger_date = trig
        if trigger_date != d or rsi_4h > cfg.athdip_rsi_4h_max:
            continue

        dc = _cut(daily_close, d)
        dh = _cut(daily_high, d)
        dl = _cut(daily_low, d)
        dv = _cut(daily_vol, d)
        if len(dc) < cfg.sma_slow + 5:
            continue
        tech = compute_technicals(
            dc,
            dh,
            dl,
            dv,
            donchian_window=cfg.donchian_window,
            sma_fast=cfg.sma_fast,
            sma_slow=cfg.sma_slow,
            avg_volume_window=cfg.avg_volume_window,
        )
        if tech is None or tech.price < cfg.min_price:
            continue
        ok, _, ath_setup, days_since = recent_multi_year_ath_setup(
            dh,
            dc,
            fresh_window=cfg.athdip_fresh_high_days,
            max_days_since_ath=cfg.athdip_max_days_since_ath,
        )
        if not ok or ath_setup is None:
            continue
        key = (ticker, d)
        if key in seen_days:
            continue
        seen_days.add(key)
        rows.append(
            {
                "ticker": ticker,
                "trigger_date": str(d),
                "rsi_4h": round(float(rsi_4h), 2),
                "pct_from_ath": round(pct_from_high(tech.price, float(ath_setup)), 2),
                "days_since_ath": days_since,
                "price": round(tech.price, 2),
            }
        )
    return rows


def summarize(variant: str, signals: list[dict], daily_map: dict[str, pd.Series]) -> VariantStats:
    if not signals:
        return VariantStats(
            variant=variant,
            n_signals=0,
            n_tickers=0,
            avg_rsi=float("nan"),
            win_5d=float("nan"),
            avg_5d=float("nan"),
            med_5d=float("nan"),
            win_10d=float("nan"),
            avg_10d=float("nan"),
            med_10d=float("nan"),
            win_20d=float("nan"),
            avg_20d=float("nan"),
            med_20d=float("nan"),
            avg_pct_ath=float("nan"),
        )

    rets = {5: [], 10: [], 20: []}
    for sig in signals:
        dc = daily_map.get(sig["ticker"])
        if dc is None or dc.empty:
            continue
        td = pd.Timestamp(sig["trigger_date"]).date()
        for h in (5, 10, 20):
            r = fwd_ret(dc, td, h)
            if r is not None:
                rets[h].append(r)

    def _stats(xs: list[float]) -> tuple[float, float, float]:
        if not xs:
            return float("nan"), float("nan"), float("nan")
        a = np.asarray(xs, dtype=float)
        return float((a > 0).mean() * 100), float(a.mean()), float(np.median(a))

    w5, a5, m5 = _stats(rets[5])
    w10, a10, m10 = _stats(rets[10])
    w20, a20, m20 = _stats(rets[20])
    return VariantStats(
        variant=variant,
        n_signals=len(signals),
        n_tickers=len({s["ticker"] for s in signals}),
        avg_rsi=float(np.mean([s["rsi_4h"] for s in signals])),
        win_5d=w5,
        avg_5d=a5,
        med_5d=m5,
        win_10d=w10,
        avg_10d=a10,
        med_10d=m10,
        win_20d=w20,
        avg_20d=a20,
        med_20d=m20,
        avg_pct_ath=float(np.mean([s["pct_from_ath"] for s in signals])),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="sp500")
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--period", default="730d", help="Intraday history window (Yahoo max ~730d for 1h)")
    p.add_argument("--rsi-max", type=float, default=31.0)
    args = p.parse_args(argv)

    cfg = ScreenerConfig(
        athdip_rsi_4h_max=args.rsi_max,
        athdip_require_multi_year_break=True,
    )
    tickers = resolve_universe(limit=args.limit, universe=args.universe)
    print(f"Universe {args.universe} limit={args.limit} → {len(tickers)} tickers", flush=True)
    print(f"Intraday period={args.period} · RSI≤{args.rsi_max}", flush=True)

    print("Downloading daily bars (ATH setup)…", flush=True)
    daily = download_prices(tickers, history_days=cfg.athdip_history_days, chunk_size=40)
    daily_close: dict[str, pd.Series] = {}
    daily_high: dict[str, pd.Series] = {}
    daily_low: dict[str, pd.Series] = {}
    daily_vol: dict[str, pd.Series] = {}
    for t in tickers:
        c = series_for(daily, "Close", t).dropna()
        h = series_for(daily, "High", t).dropna()
        lo = series_for(daily, "Low", t).dropna()
        v = series_for(daily, "Volume", t).dropna()
        if len(c) >= 100:
            daily_close[t] = c
            daily_high[t] = h
            daily_low[t] = lo if not lo.empty else c
            daily_vol[t] = v if not v.empty else pd.Series(1e6, index=c.index)
    tickers = [t for t in tickers if t in daily_close]
    print(f"Daily coverage: {len(tickers)}")

    print("Downloading native 4h (RTH)…")
    rth_4h = download_4h_native(tickers, period=args.period)
    print(f"  RTH 4h coverage: {len(rth_4h)}")

    print("Downloading 1h + prepost (extended)…")
    h1_ext = download_1h(tickers, period=args.period, prepost=True)
    print(f"  1h extended coverage: {len(h1_ext)}")

    extended_4h: dict[str, pd.Series] = {}
    calendar_4h: dict[str, pd.Series] = {}
    for t, s in h1_ext.items():
        extended_4h[t] = resample_to_4h(s, how="extended")
        calendar_4h[t] = resample_to_4h(s, how="calendar_4h")

    variants = {
        "rth": rth_4h,
        "extended": extended_4h,
        "calendar_4h": calendar_4h,
    }

    all_signals: dict[str, list[dict]] = {}
    summaries: list[VariantStats] = []
    for name, panel in variants.items():
        print(f"Scanning signals · {name}…")
        sigs: list[dict] = []
        for t in tickers:
            c4 = panel.get(t)
            if c4 is None or c4.empty:
                continue
            sigs.extend(
                collect_signals(
                    ticker=t,
                    daily_high=daily_high[t],
                    daily_low=daily_low[t],
                    daily_close=daily_close[t],
                    daily_vol=daily_vol[t],
                    close_4h=c4,
                    cfg=cfg,
                )
            )
        # de-dupe by ticker+date
        uniq = {(s["ticker"], s["trigger_date"]): s for s in sigs}
        sigs = list(uniq.values())
        all_signals[name] = sigs
        st = summarize(name, sigs, daily_close)
        summaries.append(st)
        print(
            f"  {name}: n={st.n_signals} tickers={st.n_tickers} "
            f"avgRSI={st.avg_rsi:.1f} "
            f"5d win={st.win_5d:.0f}% avg={st.avg_5d:+.2f}% | "
            f"10d win={st.win_10d:.0f}% avg={st.avg_10d:+.2f}% | "
            f"20d win={st.win_20d:.0f}% avg={st.avg_20d:+.2f}%"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "athdip_rsi_session_backtest_summary.json"
    signals_path = OUTPUT / "athdip_rsi_session_backtest_signals.csv"
    summary_path.write_text(
        json.dumps(
            {
                "as_of": datetime.now(timezone.utc).date().isoformat(),
                "universe": args.universe,
                "limit": args.limit,
                "period": args.period,
                "rsi_max": args.rsi_max,
                "note": (
                    "Yahoo US equities have no true overnight 24h bars. "
                    "calendar_4h / extended use 1h+prepost (~04:00–20:00 ET). "
                    "rth uses native Yahoo 4h (production athdip)."
                ),
                "variants": [asdict(s) for s in summaries],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Flat signal dump with variant column
    flat = []
    for name, sigs in all_signals.items():
        for s in sigs:
            flat.append({"variant": name, **s})
    if flat:
        pd.DataFrame(flat).to_csv(signals_path, index=False)
    else:
        pd.DataFrame(columns=["variant", "ticker", "trigger_date", "rsi_4h"]).to_csv(
            signals_path, index=False
        )

    print("\n=== SUMMARY ===")
    print(f"{'variant':12} {'n':>5} {'tickers':>7} {'avgRSI':>7} "
          f"{'win5':>6} {'avg5':>7} {'win10':>6} {'avg10':>7} {'win20':>6} {'avg20':>7}")
    best_10 = None
    for s in summaries:
        print(
            f"{s.variant:12} {s.n_signals:5d} {s.n_tickers:7d} {s.avg_rsi:7.1f} "
            f"{s.win_5d:5.0f}% {s.avg_5d:+6.2f}% "
            f"{s.win_10d:5.0f}% {s.avg_10d:+6.2f}% "
            f"{s.win_20d:5.0f}% {s.avg_20d:+6.2f}%"
        )
        if s.n_signals >= 5 and not np.isnan(s.avg_10d):
            if best_10 is None or s.avg_10d > best_10.avg_10d:
                best_10 = s

    print(f"\nWrote {summary_path}")
    print(f"Wrote {signals_path}")
    if best_10:
        print(
            f"\nBest by 10d avg return (n≥5): {best_10.variant} "
            f"({best_10.avg_10d:+.2f}%, win {best_10.win_10d:.0f}%, n={best_10.n_signals})"
        )
        if best_10.variant == "rth":
            print("→ Keep production athdip on Yahoo native 4h (RTH).")
        elif best_10.variant == "extended":
            print("→ Consider switching athdip RSI to extended-hours 4h (1h+prepost resample).")
        else:
            print("→ calendar_4h (extended feed) won — still not true overnight 24h.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
