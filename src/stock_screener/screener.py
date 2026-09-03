from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from stock_screener.config import ScreenerConfig
from stock_screener.data.earnings import EarningsAssessment, fetch_earnings_map
from stock_screener.data.fundamentals import (
    FundamentalsSnapshot,
    fetch_fundamentals_map,
)
from stock_screener.data.grok_x import GrokXAssessment, enrich_top_with_grok
from stock_screener.data.news import NewsAssessment, fetch_news_assessment
from stock_screener.data.prices import download_prices, series_for
from stock_screener.data.reddit import (
    RedditAssessment,
    assess_tickers_from_index,
    build_reddit_mention_index,
)
from stock_screener.data.politicians import (
    collect_politician_trades,
    score_politician_tickers,
)
from stock_screener.data.sentiment import MarketSentiment, assess_market_sentiment
from stock_screener.data.universe import resolve_universe
from stock_screener.features import TechnicalFeatures, compute_technicals
from stock_screener.scoring import ScreenRow, build_catalyst_row, build_support_row

logger = logging.getLogger(__name__)


def _rows_to_passed(rows: list[ScreenRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.to_dict() for r in rows])
    passed = df[df["pass_filters"]].copy()
    return passed.sort_values("composite", ascending=False).reset_index(drop=True)


def _fund_to_passed(snaps: dict[str, FundamentalsSnapshot]) -> pd.DataFrame:
    rows = [s.to_dict() for s in snaps.values() if s.pass_filters]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


def _politicians_to_df(scores: dict) -> pd.DataFrame:
    rows = [s.to_dict() for s in scores.values() if s.pass_filters]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _enrich_with_fundamentals(
    df: pd.DataFrame,
    fund_map: dict[str, FundamentalsSnapshot],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    scores = []
    whys = []
    pegs = []
    fcfys = []
    roes = []
    pes = []
    gross_ms = []
    profit_ms = []
    for t in out["ticker"]:
        snap = fund_map.get(t)
        if snap and snap.pass_filters:
            scores.append(snap.composite)
            whys.append(snap.why)
            pegs.append(snap.peg if snap.peg is not None else float("nan"))
            fcfys.append(snap.fcf_yield if snap.fcf_yield is not None else float("nan"))
            roes.append(snap.roe if snap.roe is not None else float("nan"))
            pes.append(snap.trailing_pe if snap.trailing_pe is not None else float("nan"))
            gross_ms.append(
                snap.gross_margin if snap.gross_margin is not None else float("nan")
            )
            profit_ms.append(
                snap.profit_margin if snap.profit_margin is not None else float("nan")
            )
        else:
            scores.append(float("nan"))
            whys.append("")
            pegs.append(float("nan"))
            fcfys.append(float("nan"))
            roes.append(float("nan"))
            pes.append(float("nan"))
            gross_ms.append(float("nan"))
            profit_ms.append(float("nan"))
    out["fund_score"] = scores
    out["fund_why"] = whys
    out["peg"] = pegs
    out["fcf_yield"] = fcfys
    out["roe"] = roes
    out["trailing_pe"] = pes
    out["gross_margin"] = gross_ms
    out["profit_margin"] = profit_ms
    return out


def _enrich_with_politicians(df: pd.DataFrame, pol_map: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    scores = []
    whys = []
    for t in out["ticker"]:
        snap = pol_map.get(t)
        if snap and snap.pass_filters:
            scores.append(snap.score)
            whys.append(snap.why)
        else:
            scores.append(float("nan"))
            whys.append("")
    out["politician_score"] = scores
    out["politician_why"] = whys
    return out


def run_screener(
    config: ScreenerConfig | None = None,
    tickers: list[str] | None = None,
    fetch_news: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, MarketSentiment]:
    """
    Returns (support_df, catalyst_df, fundamentals_df, politicians_df, market).
    """
    cfg = config or ScreenerConfig()
    mode = (cfg.mode or "all").lower()
    want_support = mode in {"support", "both", "all"}
    want_catalyst = mode in {"catalyst", "both", "all"}
    want_fundamentals = mode in {"fundamentals", "all"} or (
        cfg.enable_fundamentals and mode in {"support", "catalyst", "both", "all"}
    )
    want_politicians = mode in {"politicians", "all"} or (
        cfg.enable_politicians and mode in {"support", "catalyst", "both", "all"}
    )

    universe = resolve_universe(
        limit=cfg.max_tickers,
        tickers=tickers,
        universe=cfg.universe,
    )
    market = assess_market_sentiment(cfg)
    logger.info("Market sentiment: %s (%s)", market.label, market.reason)
    logger.info(
        "Universe size: %d (%s) · mode=%s · regime=%s",
        len(universe),
        cfg.universe if not tickers else "custom",
        mode,
        market.regime.note,
    )

    empty = pd.DataFrame()
    support_df = empty
    catalyst_df = empty
    fundamentals_df = empty
    politicians_df = empty
    liquid: list[str] = []
    tech_rows: list[tuple[str, TechnicalFeatures]] = []

    # Politicians-only can skip price download
    need_prices = want_support or want_catalyst or mode in {"fundamentals", "all"}

    if need_prices:
        prices = download_prices(
            universe,
            history_days=cfg.history_days,
            chunk_size=cfg.price_chunk_size,
        )
        if prices.empty and mode != "politicians":
            return empty, empty, empty, empty, market

        for ticker in universe:
            tech = compute_technicals(
                series_for(prices, "Close", ticker),
                series_for(prices, "High", ticker),
                series_for(prices, "Low", ticker),
                series_for(prices, "Volume", ticker),
                donchian_window=cfg.donchian_window,
                sma_fast=cfg.sma_fast,
                sma_slow=cfg.sma_slow,
                avg_volume_window=cfg.avg_volume_window,
            )
            if tech is None:
                continue
            tech_rows.append((ticker, tech))

        logger.info(
            "Computed technicals for %d / %d tickers", len(tech_rows), len(universe)
        )
        liquid = [
            t
            for t, tech in tech_rows
            if tech.price >= cfg.min_price
            and tech.avg_dollar_volume >= cfg.min_avg_dollar_volume
        ]

        news_map: dict[str, NewsAssessment] = {}
        if fetch_news and (want_support or want_catalyst):
            logger.info("Fetching Yahoo news for %d liquid candidates", len(liquid))

            def _one_news(t: str) -> tuple[str, NewsAssessment]:
                return t, fetch_news_assessment(
                    t,
                    news_hours=cfg.news_hours,
                    pause=cfg.sleep_between_news,
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(_one_news, t) for t in liquid]
                for fut in as_completed(futures):
                    t, assessment = fut.result()
                    news_map[t] = assessment

        earnings_map: dict[str, EarningsAssessment] = {}
        if cfg.enable_earnings and (want_support or want_catalyst):
            earnings_map = fetch_earnings_map(liquid)

        reddit_map: dict[str, RedditAssessment] = {}
        if cfg.enable_reddit and (want_support or want_catalyst):
            index = build_reddit_mention_index(known_tickers=set(universe))
            reddit_map = assess_tickers_from_index([t for t, _ in tech_rows], index)

        neutral_news = NewsAssessment(
            score=50.0,
            veto=False,
            headline_count=0,
            top_headline="",
            reason="News skipped",
        )
        neutral_earnings = EarningsAssessment(
            None, 50.0, "Earnings skipped", 50.0, "Earnings skipped"
        )
        neutral_reddit = RedditAssessment(50.0, 0, "Reddit skipped")

        def _build_all(
            grok_map: dict[str, GrokXAssessment],
        ) -> tuple[list[ScreenRow], list[ScreenRow]]:
            support_rows: list[ScreenRow] = []
            catalyst_rows: list[ScreenRow] = []
            for ticker, tech in tech_rows:
                news = news_map.get(ticker, neutral_news)
                earn = earnings_map.get(ticker, neutral_earnings)
                reddit = reddit_map.get(ticker, neutral_reddit)
                grok = grok_map.get(ticker)
                if want_support:
                    support_rows.append(
                        build_support_row(
                            ticker, tech, market, news, earn, reddit, grok, cfg
                        )
                    )
                if want_catalyst:
                    catalyst_rows.append(
                        build_catalyst_row(
                            ticker, tech, market, news, earn, reddit, grok, cfg
                        )
                    )
            return support_rows, catalyst_rows

        support_rows, catalyst_rows = _build_all({})
        support_df = _rows_to_passed(support_rows) if want_support else empty
        catalyst_df = _rows_to_passed(catalyst_rows) if want_catalyst else empty

        grok_map: dict[str, GrokXAssessment] = {}
        if cfg.enable_grok and (want_support or want_catalyst):
            seed: list[str] = []
            if want_support and not support_df.empty:
                seed.extend(support_df["ticker"].head(cfg.grok_top_n).tolist())
            if want_catalyst and not catalyst_df.empty:
                seed.extend(catalyst_df["ticker"].head(cfg.grok_top_n).tolist())
            seen: set[str] = set()
            top: list[str] = []
            for t in seed:
                if t not in seen:
                    seen.add(t)
                    top.append(t)
            top = top[: cfg.grok_top_n]
            grok_map = enrich_top_with_grok(top)

        if grok_map:
            support_rows, catalyst_rows = _build_all(grok_map)
            support_df = _rows_to_passed(support_rows) if want_support else empty
            catalyst_df = _rows_to_passed(catalyst_rows) if want_catalyst else empty

        if want_fundamentals and cfg.enable_fundamentals:
            fund_tickers: list[str] = []
            if mode == "fundamentals":
                fund_tickers = list(liquid)
            else:
                for df in (support_df, catalyst_df):
                    if not df.empty:
                        fund_tickers.extend(
                            df["ticker"].head(cfg.fundamentals_enrich_top).tolist()
                        )
                seen_f: set[str] = set()
                uniq: list[str] = []
                for t in fund_tickers:
                    if t not in seen_f:
                        seen_f.add(t)
                        uniq.append(t)
                fund_tickers = uniq[: max(cfg.fundamentals_enrich_top, 1)]
                if not fund_tickers:
                    fund_tickers = liquid[: cfg.fundamentals_enrich_top]

            fund_map = fetch_fundamentals_map(fund_tickers)
            fundamentals_df = _fund_to_passed(fund_map)
            if want_support and not support_df.empty:
                support_df = _enrich_with_fundamentals(support_df, fund_map)
            if want_catalyst and not catalyst_df.empty:
                catalyst_df = _enrich_with_fundamentals(catalyst_df, fund_map)

    pol_map = {}
    if want_politicians and cfg.enable_politicians:
        trades = collect_politician_trades(
            capitol_pages=cfg.politicians_capitol_pages,
            house_max_filings=cfg.politicians_house_filings,
            enable_capitol=cfg.politicians_enable_capitol,
            enable_house=cfg.politicians_enable_house,
            enable_senate=cfg.politicians_enable_senate,
        )
        pol_map = score_politician_tickers(
            trades,
            lag_cap=cfg.politicians_lag_cap_days,
            min_trades=cfg.politicians_min_trades,
        )
        politicians_df = _politicians_to_df(pol_map)
        # Optional enrich technical modes when tickers overlap
        if want_support and not support_df.empty:
            support_df = _enrich_with_politicians(support_df, pol_map)
        if want_catalyst and not catalyst_df.empty:
            catalyst_df = _enrich_with_politicians(catalyst_df, pol_map)

    return support_df, catalyst_df, fundamentals_df, politicians_df, market
