#!/usr/bin/env python3
"""EXPERIMENT ONLY — do not wire into production screener.

Compare YTD weekly top-10 EW books when ranking blends technicals with:

  1) tech_only          — current backtest-style catalyst gates (neutral social)
  2) existing_social    — Yahoo news polarity ± Reddit (live snapshot)
  3) stocktwits         — StockTwits cashtag Bullish/Bearish stream (live)
  4) x_quote_news       — Google News RSS headlines that mention X/Twitter + ticker
  5) stocktwits_only    — rank by StockTwits score alone (liquidity filter)
  6) x_quote_news_only  — rank by X-quoting news score alone

IMPORTANT CAVEAT
  StockTwits / Reddit / news RSS are **live snapshots**, not point-in-time history.
  This experiment applies today's social scores retrospectively across YTD
  rebalances → **look-ahead bias**. Use only to compare feed *shape* / ranking
  tilt vs tech-only, not as proof of live edge.

Outputs:
  output/experiment_social_feeds_compare.csv
  output/experiment_social_feeds_scores.csv
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_screener.config import ScreenerConfig
from stock_screener.data.news import NewsAssessment, fetch_news_assessment
from stock_screener.data.reddit import (
    RedditAssessment,
    assess_tickers_from_index,
    build_reddit_mention_index,
)
from stock_screener.data.universe import resolve_universe
from stock_screener.features import compute_technicals
from stock_screener.scoring import build_catalyst_row
import scripts.backtest_ytd as bt  # noqa: E402

logger = logging.getLogger("experiment_social")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

OUT = ROOT / "output"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 stock-screener-experiment/0.1"
)
_TAG = re.compile(r"<[^>]+>")
POS = {
    "beat",
    "surge",
    "rally",
    "upgrade",
    "buyback",
    "growth",
    "record",
    "bullish",
    "soar",
    "outperform",
}
NEG = {
    "miss",
    "cut",
    "downgrade",
    "lawsuit",
    "probe",
    "fraud",
    "bearish",
    "plunge",
    "layoff",
    "warning",
}


@dataclass
class SocialSnap:
    ticker: str
    stocktwits: float
    st_msgs: int
    st_bull: int
    st_bear: int
    x_news: float
    x_headlines: int
    yahoo_news: float
    reddit: float
    existing: float  # blend yahoo + reddit


def _strip(raw: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", raw or ""))).strip()


def fetch_stocktwits(ticker: str, limit: int = 30) -> tuple[float, int, int, int]:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json?limit={limit}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=12,
        )
        if resp.status_code != 200:
            return 50.0, 0, 0, 0
        msgs = (resp.json() or {}).get("messages") or []
    except Exception:  # noqa: BLE001
        return 50.0, 0, 0, 0
    bull = bear = 0
    for m in msgs[:limit]:
        ent = (m.get("entities") or {}).get("sentiment") or {}
        label = ent.get("basic") if isinstance(ent, dict) else None
        if label == "Bullish":
            bull += 1
        elif label == "Bearish":
            bear += 1
    labeled = bull + bear
    if labeled == 0:
        score = 50.0 + min(10.0, len(msgs) * 0.5)  # chatter without labels
    else:
        score = 50.0 + 45.0 * ((bull - bear) / labeled)
        score += min(5.0, labeled * 0.15)
    return float(np.clip(score, 0, 100)), len(msgs), bull, bear


def fetch_x_quote_news(ticker: str) -> tuple[float, int]:
    """Google News RSS: ticker + (Twitter|X|tweet) mentions."""
    q = f'("{ticker}" OR ${ticker}) (Twitter OR "on X" OR tweet OR tweets)'
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if resp.status_code != 200:
            return 50.0, 0
        root = ET.fromstring(resp.content)
    except Exception:  # noqa: BLE001
        return 50.0, 0
    channel = root.find("channel")
    if channel is None:
        return 50.0, 0
    titles: list[str] = []
    for item in channel.findall("item")[:20]:
        titles.append(_strip(item.findtext("title") or ""))
    if not titles:
        return 50.0, 0
    pos = neg = 0
    blob = " ".join(titles).lower()
    for w in POS:
        pos += blob.count(w)
    for w in NEG:
        neg += blob.count(w)
    total = pos + neg
    if total == 0:
        score = 50.0 + min(15.0, len(titles) * 1.5)
    else:
        score = 50.0 + 40.0 * ((pos - neg) / total) + min(10.0, len(titles))
    return float(np.clip(score, 0, 100)), len(titles)


def gather_social(tickers: list[str]) -> dict[str, SocialSnap]:
    print(f"Fetching social snapshots for {len(tickers)} tickers…")
    # Reddit once
    reddit_map: dict[str, RedditAssessment] = {}
    try:
        idx = build_reddit_mention_index(known_tickers=set(tickers))
        reddit_map = assess_tickers_from_index(tickers, idx)
        print(f"  Reddit index ok ({sum(1 for v in reddit_map.values() if v.mentions)} mentioned)")
    except Exception as exc:  # noqa: BLE001
        print(f"  Reddit skipped: {exc}")

    news_map: dict[str, NewsAssessment] = {}

    def _news(t: str) -> tuple[str, NewsAssessment]:
        return t, fetch_news_assessment(t)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_news, t) for t in tickers]
        for i, fut in enumerate(as_completed(futs), 1):
            t, n = fut.result()
            news_map[t] = n
            if i % 20 == 0:
                print(f"  Yahoo news {i}/{len(tickers)}")

    out: dict[str, SocialSnap] = {}

    def _one(t: str) -> SocialSnap:
        st, nmsg, bull, bear = fetch_stocktwits(t)
        time.sleep(0.05)  # be polite
        xn, xh = fetch_x_quote_news(t)
        time.sleep(0.05)
        yn = news_map.get(t)
        rd = reddit_map.get(t)
        yscore = yn.score if yn else 50.0
        rscore = rd.score if rd else 50.0
        existing = 0.6 * yscore + 0.4 * rscore
        return SocialSnap(
            ticker=t,
            stocktwits=st,
            st_msgs=nmsg,
            st_bull=bull,
            st_bear=bear,
            x_news=xn,
            x_headlines=xh,
            yahoo_news=yscore,
            reddit=rscore,
            existing=existing,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_one, t) for t in tickers]
        for i, fut in enumerate(as_completed(futs), 1):
            snap = fut.result()
            out[snap.ticker] = snap
            if i % 15 == 0:
                print(f"  StockTwits/X-RSS {i}/{len(tickers)}")
    return out


def tech_catalyst_score(
    prices: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    cfg: ScreenerConfig,
    market,
) -> float | None:
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


def blend(tech: float | None, social: float, w_social: float = 0.35) -> float | None:
    if tech is None:
        return None
    return (1.0 - w_social) * tech + w_social * social


def pick_book(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    universe: list[str],
    cfg: ScreenerConfig,
    market,
    social: dict[str, SocialSnap],
    variant: str,
    top_n: int,
) -> list[str]:
    scored: list[tuple[str, float]] = []
    for t in universe:
        tech = tech_catalyst_score(prices, t, as_of, cfg, market)
        snap = social.get(t)
        if variant == "tech_only":
            s = tech
        elif variant == "existing_social":
            s = blend(tech, snap.existing if snap else 50.0)
        elif variant == "stocktwits":
            s = blend(tech, snap.stocktwits if snap else 50.0)
        elif variant == "x_quote_news":
            s = blend(tech, snap.x_news if snap else 50.0)
        elif variant == "stocktwits_only":
            if tech is None:
                continue
            s = snap.stocktwits if snap else 50.0
        elif variant == "x_quote_news_only":
            if tech is None:
                continue
            s = snap.x_news if snap else 50.0
        else:
            raise ValueError(variant)
        if s is None:
            continue
        scored.append((t, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


def run_variant(
    prices: pd.DataFrame,
    universe: list[str],
    cfg: ScreenerConfig,
    social: dict[str, SocialSnap],
    variant: str,
    top_n: int,
    year_start: pd.Timestamp,
) -> bt.BacktestResult:
    close_spy = bt._col(prices, "Close", "SPY")
    vix = bt._col(prices, "Close", "^VIX")
    dates = close_spy.loc[year_start:].index
    # Fridays
    fridays = [d for d in dates if pd.Timestamp(d).weekday() == 4]
    if not fridays:
        fridays = list(dates[::5])

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    weeks = 0
    holdings_n: list[int] = []
    prev: list[str] = []
    prev_date = None

    for d in fridays:
        as_of = pd.Timestamp(d)
        market = bt.market_at(
            float(vix.loc[:as_of].iloc[-1]) if len(vix.loc[:as_of]) else None,
            cfg,
        )
        book = pick_book(prices, as_of, universe, cfg, market, social, variant, top_n)
        if prev and prev_date is not None:
            rets = []
            for t in prev:
                c = bt._col(prices, "Close", t)
                seg = c.loc[prev_date:as_of]
                if len(seg) >= 2 and seg.iloc[0] > 0:
                    rets.append(float(seg.iloc[-1] / seg.iloc[0] - 1.0))
            if rets:
                r = float(np.mean(rets))
                equity *= 1.0 + r
                weeks += 1
                if r > 0:
                    wins += 1
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)
        holdings_n.append(len(book))
        prev = book
        prev_date = as_of

    # mark last book to end
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
                equity *= 1.0 + r
                weeks += 1
                if r > 0:
                    wins += 1
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)

    spy_seg = close_spy.loc[year_start:]
    spy_ret = float(spy_seg.iloc[-1] / spy_seg.iloc[0] - 1.0) * 100.0 if len(spy_seg) > 1 else 0.0
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
    print("EXPERIMENT: social feed comparison (NOT production)")
    print("Live social snapshots applied retrospectively → look-ahead bias")
    print("=" * 72)

    cfg = ScreenerConfig(
        universe="sp500",
        max_tickers=80,
        require_above_sma_slow=False,
        enable_reddit=False,
        enable_grok=False,
        enable_fundamentals=False,
        enable_politicians=False,
        enable_trump=False,
    )
    universe = resolve_universe(limit=80, universe="sp500")
    today = datetime.now(timezone.utc).date()
    year_start = pd.Timestamp(f"{today.year}-01-01")
    warmup = (year_start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    social = gather_social(universe)
    OUT.mkdir(parents=True, exist_ok=True)
    scores_path = OUT / "experiment_social_feeds_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "stocktwits",
                "st_msgs",
                "st_bull",
                "st_bear",
                "x_news",
                "x_headlines",
                "yahoo_news",
                "reddit",
                "existing",
            ],
        )
        w.writeheader()
        for t in universe:
            s = social[t]
            w.writerow(
                {
                    "ticker": t,
                    "stocktwits": round(s.stocktwits, 2),
                    "st_msgs": s.st_msgs,
                    "st_bull": s.st_bull,
                    "st_bear": s.st_bear,
                    "x_news": round(s.x_news, 2),
                    "x_headlines": s.x_headlines,
                    "yahoo_news": round(s.yahoo_news, 2),
                    "reddit": round(s.reddit, 2),
                    "existing": round(s.existing, 2),
                }
            )
    print(f"Wrote {scores_path}")

    # Coverage summary
    st_cov = sum(1 for s in social.values() if s.st_msgs > 0)
    x_cov = sum(1 for s in social.values() if s.x_headlines > 0)
    print(f"Coverage: StockTwits msgs>0: {st_cov}/{len(universe)} · X-news headlines>0: {x_cov}/{len(universe)}")

    print(f"Downloading prices ({len(universe)} + SPY/VIX) {warmup} → {end}…")
    prices = bt.download_panel(universe, warmup, end)

    variants = [
        "tech_only",
        "existing_social",
        "stocktwits",
        "x_quote_news",
        "stocktwits_only",
        "x_quote_news_only",
    ]
    results = []
    for v in variants:
        print(f"Backtesting {v}…")
        r = run_variant(prices, universe, cfg, social, v, top_n=10, year_start=year_start)
        results.append(r)
        print(
            f"  {v:20} ret={r.ytd_return_pct:+.2f}% excess={r.excess_pct:+.2f}% "
            f"win={r.win_rate_pct:.1f}% dd={r.max_drawdown_pct:.2f}% avg#={r.avg_holdings}"
        )

    cmp_path = OUT / "experiment_social_feeds_compare.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "return_pct",
                "spy_pct",
                "excess_pct",
                "win_pct",
                "max_dd_pct",
                "avg_holdings",
                "rebalances",
                "caveat",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "variant": r.mode,
                    "return_pct": r.ytd_return_pct,
                    "spy_pct": r.spy_return_pct,
                    "excess_pct": r.excess_pct,
                    "win_pct": r.win_rate_pct,
                    "max_dd_pct": r.max_drawdown_pct,
                    "avg_holdings": r.avg_holdings,
                    "rebalances": r.rebalance_count,
                    "caveat": "live_social_lookbehind",
                }
            )
    print(f"\nWrote {cmp_path}")
    print("\n=== COMPARISON (YTD weekly top-10 EW, SP500/80) ===")
    print(f"{'variant':20} {'ret%':>8} {'SPY%':>8} {'excess':>8} {'win%':>7} {'maxDD':>8}")
    for r in sorted(results, key=lambda x: x.excess_pct, reverse=True):
        print(
            f"{r.mode:20} {r.ytd_return_pct:+8.2f} {r.spy_return_pct:+8.2f} "
            f"{r.excess_pct:+8.2f} {r.win_rate_pct:6.1f}% {r.max_drawdown_pct:8.2f}"
        )
    print(
        "\nNot production. Do not merge StockTwits / X-RSS into screener from this run alone."
    )


if __name__ == "__main__":
    main()
