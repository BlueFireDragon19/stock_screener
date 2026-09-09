#!/usr/bin/env python3
"""EXPERIMENT ONLY — do not wire into production screener.

Compare YTD weekly top-10 EW books when ranking / sizing uses prediction markets:

  1) tech_only           — catalyst technicals only (baseline)
  2) pm_ticker           — blend tech with Polymarket per-ticker odds
  3) pm_ticker_only      — rank by Polymarket ticker score (liquid filter)
  4) pm_macro_size       — tech rank; shrink/expand book via PM macro risk
  5) kalshi_macro_size   — tech rank; book size via Kalshi Fed/rate-cut odds
  6) pm_both             — Polymarket ticker blend + PM macro book size

IMPORTANT CAVEAT
  Polymarket / Kalshi odds are **live snapshots**, not point-in-time history.
  Applied retrospectively across YTD → **look-ahead bias**. Compare feed shape
  only; not proof of live edge.

Outputs:
  output/experiment_prediction_markets_compare.csv
  output/experiment_prediction_markets_scores.csv
  output/experiment_prediction_markets_macro.json
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_screener.config import ScreenerConfig
from stock_screener.data.universe import resolve_universe
from stock_screener.features import compute_technicals
from stock_screener.scoring import build_catalyst_row
import scripts.backtest_ytd as bt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("experiment_pm")

OUT = ROOT / "output"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "stock-screener-experiment/0.1"
    )
}
PM_SEARCH = "https://gamma-api.polymarket.com/public-search"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Mega-cap aliases help Polymarket search hit company-named markets.
ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("NVIDIA", "Nvidia"),
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "GOOGL": ("Google", "Alphabet"),
    "GOOG": ("Google", "Alphabet"),
    "AMZN": ("Amazon",),
    "META": ("Meta", "Facebook"),
    "TSLA": ("Tesla",),
    "AMD": ("AMD", "Advanced Micro Devices"),
    "AVGO": ("Broadcom",),
    "NFLX": ("Netflix",),
}


@dataclass
class TickerPM:
    ticker: str
    score: float
    n_markets: int
    volume: float
    top_question: str


@dataclass
class MacroSnap:
    source: str
    risk_on_score: float  # 0–100 (higher = risk-on)
    recession_yes: float | None
    detail: str


def _parse_json_field(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def _yes_prob(outcomes, prices) -> float | None:
    outcomes = _parse_json_field(outcomes) or []
    prices = _parse_json_field(prices) or []
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    try:
        probs = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() in {"yes", "up"}:
            return probs[i]
    return probs[0]


def _outcome_prob(outcomes, prices, needle: str) -> float | None:
    outcomes = _parse_json_field(outcomes) or []
    prices = _parse_json_field(prices) or []
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    needle_l = needle.lower()
    try:
        probs = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    for i, o in enumerate(outcomes):
        if needle_l in str(o).lower():
            return probs[i]
    return None


def _bullish_from_market(ticker: str, question: str, outcomes, prices) -> float | None:
    """Map a market to 0–100 bullishness for ticker (None = skip)."""
    q = (question or "").lower()
    t = ticker.lower()
    aliases = (t,) + tuple(a.lower() for a in ALIASES.get(ticker.upper(), ()))
    if not any(a in q for a in aliases):
        # Still allow if an outcome names the company
        named = False
        for a in aliases:
            if _outcome_prob(outcomes, prices, a) is not None:
                named = True
                break
        if not named:
            return None

    # Multi-outcome "Google or NVIDIA worth more"
    for a in aliases:
        p = _outcome_prob(outcomes, prices, a)
        if p is not None and ("or" in q or "worth more" in q or "market cap" in q):
            return 100.0 * p

    yes = _yes_prob(outcomes, prices)
    if yes is None:
        return None

    if "up or down" in q or re.search(r"\bup\b.*\bdown\b", q):
        return 100.0 * yes  # Up probability
    if "(low)" in q or " hit (low)" in q or "close below" in q or "fall below" in q:
        return 100.0 * (1.0 - yes)  # high P(low) = bearish
    if (
        "(high)" in q
        or " hit (high)" in q
        or "close above" in q
        or ("finish" in q and "above" in q)
        or "reach $" in q
        or "hit $" in q
    ):
        return 100.0 * yes
    if any(w in q for w in ("bankrupt", "delist", "fraud", "crash", "plunge")):
        return 100.0 * (1.0 - yes)
    # Default: treat Yes as mildly bullish mention
    return 50.0 + 40.0 * (yes - 0.5)


def fetch_pm_ticker(ticker: str, min_vol: float = 500.0) -> TickerPM:
    queries = [ticker] + list(ALIASES.get(ticker.upper(), ()))
    seen: set[str] = set()
    weighted: list[tuple[float, float]] = []  # score, vol
    top_q = ""
    top_vol = 0.0

    for q in queries:
        try:
            resp = requests.get(
                PM_SEARCH, params={"q": q}, headers=UA, timeout=20
            )
            if resp.status_code != 200:
                continue
            events = resp.json().get("events") or []
        except Exception:  # noqa: BLE001
            continue
        for ev in events:
            for m in ev.get("markets") or []:
                if m.get("closed"):
                    continue
                mid = str(m.get("id") or m.get("conditionId") or m.get("question"))
                if mid in seen:
                    continue
                seen.add(mid)
                vol = float(m.get("volumeNum") or m.get("volume") or 0.0)
                if vol < min_vol:
                    continue
                question = m.get("question") or ev.get("title") or ""
                score = _bullish_from_market(
                    ticker, question, m.get("outcomes"), m.get("outcomePrices")
                )
                if score is None:
                    continue
                weighted.append((float(np.clip(score, 0, 100)), vol))
                if vol > top_vol:
                    top_vol = vol
                    top_q = question[:140]
        time.sleep(0.05)

    if not weighted:
        return TickerPM(ticker, 50.0, 0, 0.0, "")
    vols = np.array([w for _, w in weighted], dtype=float)
    scores = np.array([s for s, _ in weighted], dtype=float)
    w = vols / vols.sum()
    return TickerPM(
        ticker=ticker,
        score=float(np.clip(scores @ w, 0, 100)),
        n_markets=len(weighted),
        volume=float(vols.sum()),
        top_question=top_q,
    )


def fetch_pm_macro() -> MacroSnap:
    """Risk-on from recession + Fed-cut expectations on Polymarket."""
    recession_yes = None
    detail_parts: list[str] = []

    # Recession
    try:
        resp = requests.get(
            PM_SEARCH, params={"q": "US recession 2026"}, headers=UA, timeout=20
        )
        for ev in resp.json().get("events") or []:
            title = (ev.get("title") or "").lower()
            if "recession" not in title:
                continue
            if "us" not in title and "united" not in title and "u.s" not in title:
                continue
            for m in ev.get("markets") or []:
                if m.get("closed"):
                    continue
                yes = _yes_prob(m.get("outcomes"), m.get("outcomePrices"))
                if yes is None:
                    continue
                recession_yes = yes
                detail_parts.append(f"recession_yes={yes:.3f}")
                break
            if recession_yes is not None:
                break
    except Exception as exc:  # noqa: BLE001
        detail_parts.append(f"recession_err={exc}")

    # Expected Fed cuts (probability-weighted count)
    expected_cuts = None
    try:
        resp = requests.get(
            PM_SEARCH, params={"q": "How many Fed rate cuts in 2026"}, headers=UA, timeout=20
        )
        for ev in resp.json().get("events") or []:
            title = (ev.get("title") or "").lower()
            if "fed" not in title or "cut" not in title:
                continue
            markets = [m for m in (ev.get("markets") or []) if not m.get("closed")]
            if len(markets) < 2:
                continue
            exp = 0.0
            mass = 0.0
            for m in markets:
                label = (m.get("groupItemTitle") or m.get("question") or "").strip()
                yes = _yes_prob(m.get("outcomes"), m.get("outcomePrices"))
                if yes is None:
                    continue
                m_cuts = re.match(r"^(\d+)", label)
                if not m_cuts:
                    continue
                n = int(m_cuts.group(1))
                exp += n * yes
                mass += yes
            if mass > 0.3:
                expected_cuts = exp
                detail_parts.append(f"fed_expected_cuts={exp:.2f}")
                break
    except Exception as exc:  # noqa: BLE001
        detail_parts.append(f"fed_err={exc}")

    # Map to risk-on 0–100
    score = 50.0
    if recession_yes is not None:
        # 0% recession → 75; 100% → 25
        score = 25.0 + 50.0 * (1.0 - float(recession_yes))
    if expected_cuts is not None:
        # 0 cuts → -10; ~2 cuts → +5; 4+ → +15
        score += float(np.clip((expected_cuts - 1.0) * 5.0, -10.0, 15.0))

    return MacroSnap(
        source="polymarket",
        risk_on_score=float(np.clip(score, 0, 100)),
        recession_yes=recession_yes,
        detail="; ".join(detail_parts) or "no macro",
    )


def fetch_kalshi_macro() -> MacroSnap:
    """Risk-on from Kalshi Fed funds / rate-cut series (live)."""
    detail: list[str] = []
    score = 50.0

    # Rate cut markets if present
    try:
        resp = requests.get(
            f"{KALSHI}/markets",
            params={"series_ticker": "KXRATECUT", "status": "open", "limit": 50},
            headers=UA,
            timeout=20,
        )
        markets = resp.json().get("markets") or []
        if markets:
            # Higher last price on "cut" outcomes → risk-on
            prices = []
            for m in markets:
                raw = m.get("last_price_dollars") or m.get("yes_bid_dollars")
                if raw is None:
                    continue
                prices.append(float(raw))
            if prices:
                p = float(np.mean(prices))
                score = 40.0 + 40.0 * p  # rough
                detail.append(f"kxratecut_avg_yes={p:.3f} n={len(prices)}")
    except Exception as exc:  # noqa: BLE001
        detail.append(f"kxratecut_err={exc}")

    # Fed funds: P(above high strikes) falling = easing = risk-on
    try:
        resp = requests.get(
            f"{KALSHI}/markets",
            params={"series_ticker": "KXFED", "status": "open", "limit": 40},
            headers=UA,
            timeout=20,
        )
        markets = resp.json().get("markets") or []
        # Prefer near-term event
        by_event: dict[str, list] = {}
        for m in markets:
            by_event.setdefault(m.get("event_ticker") or "", []).append(m)
        if by_event:
            event = sorted(by_event.keys())[0]
            ms = by_event[event]
            # Probability mass on high rate floors
            high = []
            low = []
            for m in ms:
                strike = m.get("floor_strike")
                raw = m.get("last_price_dollars") or m.get("previous_price_dollars")
                if strike is None or raw is None:
                    continue
                p = float(raw)
                if float(strike) >= 4.0:
                    high.append(p)
                if float(strike) <= 3.0:
                    low.append(p)
            if high:
                # High P(funds still elevated) = hawkish = risk-off
                hawk = float(np.mean(high))
                score = 0.6 * score + 0.4 * (100.0 * (1.0 - hawk))
                detail.append(f"kxfed_{event}_p_above_4={hawk:.3f}")
            if low:
                detail.append(f"kxfed_p_above_low={float(np.mean(low)):.3f}")
    except Exception as exc:  # noqa: BLE001
        detail.append(f"kxfed_err={exc}")

    return MacroSnap(
        source="kalshi",
        risk_on_score=float(np.clip(score, 0, 100)),
        recession_yes=None,
        detail="; ".join(detail) or "no kalshi macro",
    )


def book_size(top_n: int, risk_on: float) -> int:
    """Risk-off → fewer names; risk-on → full book."""
    if risk_on >= 60:
        return top_n
    if risk_on >= 45:
        return max(5, int(round(top_n * 0.8)))
    if risk_on >= 30:
        return max(4, int(round(top_n * 0.6)))
    return max(3, int(round(top_n * 0.4)))


def tech_catalyst_score(prices, ticker, as_of, cfg, market) -> float | None:
    close = bt._col(prices, "Close", ticker).loc[:as_of]
    high = bt._col(prices, "High", ticker).loc[:as_of]
    low = bt._col(prices, "Low", ticker).loc[:as_of]
    vol = bt._col(prices, "Volume", ticker).loc[:as_of]
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
        return None
    if tech.price < cfg.min_price or tech.avg_dollar_volume < cfg.min_avg_dollar_volume:
        return None
    row = build_catalyst_row(
        ticker,
        tech,
        market,
        bt.NEUTRAL_NEWS,
        bt.NEUTRAL_EARN,
        bt.NEUTRAL_REDDIT,
        None,
        cfg,
    )
    if not row.pass_filters or np.isnan(row.composite):
        return None
    return float(row.composite)


def blend(tech: float | None, social: float, w: float = 0.35) -> float | None:
    if tech is None:
        return None
    return (1.0 - w) * tech + w * social


def pick_book(
    prices,
    as_of,
    universe,
    cfg,
    market,
    pm: dict[str, TickerPM],
    variant: str,
    top_n: int,
    pm_macro: MacroSnap,
    kalshi_macro: MacroSnap,
) -> list[str]:
    scored: list[tuple[str, float]] = []
    for t in universe:
        tech = tech_catalyst_score(prices, t, as_of, cfg, market)
        snap = pm.get(t)
        pms = snap.score if snap else 50.0
        if variant == "tech_only":
            s = tech
        elif variant == "pm_ticker":
            s = blend(tech, pms)
        elif variant == "pm_ticker_only":
            if tech is None:
                continue
            s = pms
        elif variant in {"pm_macro_size", "kalshi_macro_size"}:
            s = tech
        elif variant == "pm_both":
            s = blend(tech, pms)
        else:
            raise ValueError(variant)
        if s is None:
            continue
        scored.append((t, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    n = top_n
    if variant == "pm_macro_size" or variant == "pm_both":
        n = book_size(top_n, pm_macro.risk_on_score)
    elif variant == "kalshi_macro_size":
        n = book_size(top_n, kalshi_macro.risk_on_score)
    return [t for t, _ in scored[:n]]


def run_variant(
    prices,
    universe,
    cfg,
    pm: dict[str, TickerPM],
    variant: str,
    top_n: int,
    year_start,
    pm_macro: MacroSnap,
    kalshi_macro: MacroSnap,
) -> bt.BacktestResult:
    close_spy = bt._col(prices, "Close", "SPY")
    vix = bt._col(prices, "Close", "^VIX")
    dates = close_spy.loc[year_start:].index
    fridays = [d for d in dates if pd.Timestamp(d).weekday() == 4] or list(dates[::5])

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = weeks = 0
    holdings_n: list[int] = []
    prev: list[str] = []
    prev_date = None

    for d in fridays:
        as_of = pd.Timestamp(d)
        market = bt.market_at(
            float(vix.loc[:as_of].iloc[-1]) if len(vix.loc[:as_of]) else None,
            cfg,
        )
        book = pick_book(
            prices,
            as_of,
            universe,
            cfg,
            market,
            pm,
            variant,
            top_n,
            pm_macro,
            kalshi_macro,
        )
        if prev and prev_date is not None:
            rets = []
            for t in prev:
                c = bt._col(prices, "Close", t)
                seg = c.loc[prev_date:as_of]
                if len(seg) >= 2 and seg.iloc[0] > 0:
                    rets.append(float(seg.iloc[-1] / seg.iloc[0] - 1.0))
            if rets:
                r = float(np.mean(rets))
                # Cash only for intentional macro shrink vs full top_n (not filter thinness).
                if variant in {"pm_macro_size", "pm_both"}:
                    r *= book_size(top_n, pm_macro.risk_on_score) / float(top_n)
                elif variant == "kalshi_macro_size":
                    r *= book_size(top_n, kalshi_macro.risk_on_score) / float(top_n)
            else:
                r = 0.0
            equity *= 1.0 + r
            weeks += 1
            if r > 0:
                wins += 1
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
        holdings_n.append(len(book))
        prev = book
        prev_date = as_of

    if prev and prev_date is not None and len(dates):
        end = pd.Timestamp(dates[-1])
        if end > prev_date:
            rets = []
            for t in prev:
                c = bt._col(prices, "Close", t)
                seg = c.loc[prev_date:end]
                if len(seg) >= 2 and seg.iloc[0] > 0:
                    rets.append(float(seg.iloc[-1] / seg.iloc[0] - 1.0))
            if rets:
                r = float(np.mean(rets))
                if variant in {"pm_macro_size", "pm_both"}:
                    r *= book_size(top_n, pm_macro.risk_on_score) / float(top_n)
                elif variant == "kalshi_macro_size":
                    r *= book_size(top_n, kalshi_macro.risk_on_score) / float(top_n)
                equity *= 1.0 + r
                weeks += 1
                if r > 0:
                    wins += 1
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)

    spy_seg = close_spy.loc[year_start:]
    spy_ret = (
        float(spy_seg.iloc[-1] / spy_seg.iloc[0] - 1.0) * 100.0 if len(spy_seg) > 1 else 0.0
    )
    strat = (equity - 1.0) * 100.0
    return bt.BacktestResult(
        mode=variant,
        ytd_return_pct=round(strat, 2),
        spy_return_pct=round(spy_ret, 2),
        excess_pct=round(strat - spy_ret, 2),
        rebalance_count=weeks,
        avg_holdings=round(float(np.mean(holdings_n)) if holdings_n else 0.0, 1),
        win_rate_pct=round(100.0 * wins / weeks, 1) if weeks else 0.0,
        max_drawdown_pct=round(max_dd * 100.0, 2),
    )


def main() -> None:
    print("=" * 72)
    print("EXPERIMENT: prediction markets (NOT production)")
    print("Live Polymarket/Kalshi odds applied retrospectively → look-ahead bias")
    print("=" * 72)

    cfg = ScreenerConfig(
        universe="sp500",
        max_tickers=80,
        enable_reddit=False,
        enable_grok=False,
        enable_x_quote_news=False,
    )
    universe = resolve_universe(limit=cfg.max_tickers, universe=cfg.universe)
    print(f"Universe: {len(universe)} tickers")

    print("Fetching Polymarket macro…")
    pm_macro = fetch_pm_macro()
    print(f"  PM macro risk_on={pm_macro.risk_on_score:.1f} | {pm_macro.detail}")

    print("Fetching Kalshi macro…")
    kalshi_macro = fetch_kalshi_macro()
    print(
        f"  Kalshi macro risk_on={kalshi_macro.risk_on_score:.1f} | {kalshi_macro.detail}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "experiment_prediction_markets_macro.json").write_text(
        json.dumps(
            {
                "polymarket": {
                    "risk_on_score": pm_macro.risk_on_score,
                    "recession_yes": pm_macro.recession_yes,
                    "detail": pm_macro.detail,
                },
                "kalshi": {
                    "risk_on_score": kalshi_macro.risk_on_score,
                    "detail": kalshi_macro.detail,
                },
                "book_size_pm": book_size(10, pm_macro.risk_on_score),
                "book_size_kalshi": book_size(10, kalshi_macro.risk_on_score),
            },
            indent=2,
        )
    )

    print(f"Fetching Polymarket ticker markets for {len(universe)} names…")
    pm: dict[str, TickerPM] = {}

    def _one(t: str) -> TickerPM:
        return fetch_pm_ticker(t)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_one, t) for t in universe]
        for i, fut in enumerate(as_completed(futs), 1):
            snap = fut.result()
            pm[snap.ticker] = snap
            if i % 20 == 0:
                print(f"  Polymarket tickers {i}/{len(universe)}")

    covered = sum(1 for s in pm.values() if s.n_markets > 0)
    print(f"Coverage: {covered}/{len(universe)} tickers with ≥1 open PM market")

    scores_path = OUT / "experiment_prediction_markets_scores.csv"
    with scores_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ticker", "pm_score", "n_markets", "volume", "top_question"],
        )
        w.writeheader()
        for t in universe:
            s = pm[t]
            w.writerow(
                {
                    "ticker": t,
                    "pm_score": round(s.score, 2),
                    "n_markets": s.n_markets,
                    "volume": round(s.volume, 1),
                    "top_question": s.top_question,
                }
            )
    print(f"Wrote {scores_path}")

    print("Downloading prices…")
    today = datetime.now(timezone.utc).date()
    year_start = pd.Timestamp(f"{today.year}-01-01")
    warmup = (year_start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  panel {warmup} → {end}")
    prices = bt.download_panel(universe, warmup, end)

    variants = [
        "tech_only",
        "pm_ticker",
        "pm_ticker_only",
        "pm_macro_size",
        "kalshi_macro_size",
        "pm_both",
    ]
    rows = []
    for v in variants:
        print(f"Backtesting {v}…")
        res = run_variant(
            prices,
            universe,
            cfg,
            pm,
            v,
            top_n=10,
            year_start=year_start,
            pm_macro=pm_macro,
            kalshi_macro=kalshi_macro,
        )
        print(
            f"  {v:20s} ret={res.ytd_return_pct:+.2f}% excess={res.excess_pct:+.2f}% "
            f"win={res.win_rate_pct}% dd={res.max_drawdown_pct}% avg#={res.avg_holdings}"
        )
        rows.append(res)

    out_csv = OUT / "experiment_prediction_markets_compare.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "ytd_return_pct",
                "spy_return_pct",
                "excess_pct",
                "win_rate_pct",
                "max_drawdown_pct",
                "avg_holdings",
                "rebalance_count",
            ],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.ytd_return_pct, reverse=True):
            w.writerow(
                {
                    "variant": r.mode,
                    "ytd_return_pct": r.ytd_return_pct,
                    "spy_return_pct": r.spy_return_pct,
                    "excess_pct": r.excess_pct,
                    "win_rate_pct": r.win_rate_pct,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "avg_holdings": r.avg_holdings,
                    "rebalance_count": r.rebalance_count,
                }
            )
    print(f"\nWrote {out_csv}")
    print("\n=== COMPARISON (YTD weekly top-10 EW, SP500/80) ===")
    print(f"{'variant':22s} {'ret%':>8} {'SPY%':>8} {'excess':>8} {'win%':>7} {'maxDD':>8} {'avg#':>5}")
    for r in sorted(rows, key=lambda x: x.ytd_return_pct, reverse=True):
        print(
            f"{r.mode:22s} {r.ytd_return_pct:+8.2f} {r.spy_return_pct:+8.2f} "
            f"{r.excess_pct:+8.2f} {r.win_rate_pct:6.1f}% {r.max_drawdown_pct:8.2f} "
            f"{r.avg_holdings:5.1f}"
        )
    print("\nNot production. Do not merge Polymarket/Kalshi from this run alone.")


if __name__ == "__main__":
    main()
