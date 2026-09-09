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
from stock_screener.data.x_quote_news import (
    blend_yahoo_and_x_quote,
    fetch_x_quote_news_assessment,
)
from stock_screener.data.polymarket import (
    PolymarketAssessment,
    fetch_polymarket_assessment,
)
from stock_screener.data.prices import download_intraday, download_prices, series_for
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
from stock_screener.features import (
    TechnicalFeatures,
    all_time_high,
    compute_technicals,
    min_rsi_lookback_detail,
    pct_from_high,
    recent_multi_year_ath_setup,
)
from stock_screener.scoring import (
    ScreenRow,
    build_athdip_row,
    build_catalyst_row,
    build_support_row,
)
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


def _trump_to_df(scores: dict) -> pd.DataFrame:
    rows = [s.to_dict() for s in scores.values() if s.pass_filters]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _enrich_with_trump(df: pd.DataFrame, trump_map: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    scores = []
    whys = []
    for t in out["ticker"]:
        snap = trump_map.get(t)
        if snap and snap.pass_filters:
            scores.append(snap.score)
            whys.append(snap.why)
        else:
            scores.append(float("nan"))
            whys.append("")
    out["trump_score"] = scores
    out["trump_why"] = whys
    return out


def run_screener(
    config: ScreenerConfig | None = None,
    tickers: list[str] | None = None,
    fetch_news: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    MarketSentiment,
]:
    """
    Returns (support, catalyst, fundamentals, politicians, trump, athdip, market).
    """
    cfg = config or ScreenerConfig()
    mode = (cfg.mode or "all").lower()
    want_support = mode in {"support", "both", "all"}
    want_catalyst = mode in {"catalyst", "both", "all"}
    want_athdip = mode in {"athdip", "all"}
    want_fundamentals = mode in {"fundamentals", "all"} or (
        cfg.enable_fundamentals and mode in {"support", "catalyst", "both", "all"}
    )
    want_politicians = mode in {"politicians", "all"} or (
        cfg.enable_politicians and mode in {"support", "catalyst", "both", "all"}
    )
    want_trump = mode in {"trump", "all"} or (
        cfg.enable_trump and mode in {"support", "catalyst", "both", "all"}
    )

    # Tracker-only modes don't need a price universe
    if mode in {"trump", "politicians"} and not (
        want_support or want_catalyst or want_athdip
    ):
        universe = tickers or []
    else:
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
    trump_df = empty
    athdip_df = empty
    liquid: list[str] = []
    tech_rows: list[tuple[str, TechnicalFeatures]] = []
    high_by_ticker: dict[str, pd.Series] = {}
    close_by_ticker: dict[str, pd.Series] = {}

    need_prices = (
        want_support
        or want_catalyst
        or want_athdip
        or mode in {"fundamentals", "all"}
    )
    hist_days = cfg.history_days
    if want_athdip:
        hist_days = max(hist_days, cfg.athdip_history_days)

    if need_prices:
        prices = download_prices(
            universe,
            history_days=hist_days,
            chunk_size=cfg.price_chunk_size,
        )
        if prices.empty and mode not in {"politicians", "trump"}:
            return empty, empty, empty, empty, empty, empty, market

        for ticker in universe:
            close = series_for(prices, "Close", ticker)
            high = series_for(prices, "High", ticker)
            low = series_for(prices, "Low", ticker)
            vol = series_for(prices, "Volume", ticker)
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
            tech_rows.append((ticker, tech))
            high_by_ticker[ticker] = high
            close_by_ticker[ticker] = close

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

            if cfg.enable_x_quote_news:
                logger.info(
                    "Fetching X-quoting news RSS for %d liquid candidates", len(liquid)
                )

                def _one_x(t: str) -> tuple[str, NewsAssessment]:
                    return t, fetch_x_quote_news_assessment(
                        t,
                        news_hours=cfg.news_hours,
                        pause=min(0.08, cfg.sleep_between_news),
                    )

                x_map: dict[str, NewsAssessment] = {}
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(_one_x, t) for t in liquid]
                    for fut in as_completed(futures):
                        t, assessment = fut.result()
                        x_map[t] = assessment
                for t in liquid:
                    news_map[t] = blend_yahoo_and_x_quote(
                        news_map.get(
                            t,
                            NewsAssessment(
                                50.0, False, 0, "", "News skipped"
                            ),
                        ),
                        x_map.get(t),
                        x_weight=cfg.x_quote_news_weight,
                    )

        earnings_map: dict[str, EarningsAssessment] = {}
        if cfg.enable_earnings and (want_support or want_catalyst):
            earnings_map = fetch_earnings_map(liquid)

        reddit_map: dict[str, RedditAssessment] = {}
        if cfg.enable_reddit and (want_support or want_catalyst):
            index = build_reddit_mention_index(known_tickers=set(universe))
            reddit_map = assess_tickers_from_index([t for t, _ in tech_rows], index)

        polymarket_map: dict[str, PolymarketAssessment] = {}
        if cfg.enable_polymarket and (want_support or want_catalyst):
            logger.info(
                "Fetching Polymarket odds for %d liquid candidates", len(liquid)
            )

            def _one_pm(t: str) -> tuple[str, PolymarketAssessment]:
                return t, fetch_polymarket_assessment(
                    t,
                    min_vol=cfg.polymarket_min_volume,
                    pause=min(0.05, cfg.sleep_between_news),
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(_one_pm, t) for t in liquid]
                for fut in as_completed(futures):
                    t, assessment = fut.result()
                    polymarket_map[t] = assessment
            covered = sum(1 for a in polymarket_map.values() if a.used)
            logger.info("Polymarket coverage: %d / %d liquid", covered, len(liquid))

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
                pm = polymarket_map.get(ticker)
                grok = grok_map.get(ticker)
                if want_support:
                    support_rows.append(
                        build_support_row(
                            ticker,
                            tech,
                            market,
                            news,
                            earn,
                            reddit,
                            grok,
                            cfg,
                            polymarket=pm,
                        )
                    )
                if want_catalyst:
                    catalyst_rows.append(
                        build_catalyst_row(
                            ticker,
                            tech,
                            market,
                            news,
                            earn,
                            reddit,
                            grok,
                            cfg,
                            polymarket=pm,
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

        # --- Athdip: near ATH + 4h RSI ~30 ---
        if want_athdip:
            athdip_df = _run_athdip(
                tech_rows=tech_rows,
                high_by_ticker=high_by_ticker,
                close_by_ticker=close_by_ticker,
                market=market,
                cfg=cfg,
            )

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
        if want_support and not support_df.empty:
            support_df = _enrich_with_politicians(support_df, pol_map)
        if want_catalyst and not catalyst_df.empty:
            catalyst_df = _enrich_with_politicians(catalyst_df, pol_map)

    trump_map = {}
    if want_trump and cfg.enable_trump:
        from stock_screener.data.trump_tracker import run_trump_tracker

        _posts, trump_map = run_trump_tracker(
            enable_truth=cfg.trump_enable_truth,
            enable_news=cfg.trump_enable_news,
            enable_wh=cfg.trump_enable_wh,
            enable_x=cfg.trump_enable_x and cfg.enable_grok,
            truth_max=cfg.trump_truth_max_posts,
        )
        trump_df = _trump_to_df(trump_map)
        if want_support and not support_df.empty:
            support_df = _enrich_with_trump(support_df, trump_map)
        if want_catalyst and not catalyst_df.empty:
            catalyst_df = _enrich_with_trump(catalyst_df, trump_map)

    return (
        support_df,
        catalyst_df,
        fundamentals_df,
        politicians_df,
        trump_df,
        athdip_df,
        market,
    )


def _run_athdip(
    *,
    tech_rows: list[tuple[str, TechnicalFeatures]],
    high_by_ticker: dict[str, pd.Series],
    close_by_ticker: dict[str, pd.Series],
    market: MarketSentiment,
    cfg: ScreenerConfig,
) -> pd.DataFrame:
    """Watch recent multi-year ATH setups; trigger on 4h RSI ≤ max."""
    # ticker, tech, ath, pct_ath, prior_high, days_since_ath, setup_ok
    candidates: list[
        tuple[str, TechnicalFeatures, float, float, float | None, int | None, bool]
    ] = []
    for ticker, tech in tech_rows:
        if tech.price < cfg.min_price:
            continue
        if tech.avg_dollar_volume < cfg.min_avg_dollar_volume:
            continue
        if cfg.athdip_require_uptrend:
            if cfg.athdip_require_sma50:
                if not (tech.above_sma_fast and tech.above_sma_slow):
                    continue
            elif not tech.above_sma_slow:
                continue
        if cfg.athdip_drop_death_cross and tech.death_cross:
            continue
        high = high_by_ticker.get(ticker, pd.Series(dtype=float))
        close = close_by_ticker.get(ticker, pd.Series(dtype=float))
        ath = all_time_high(high)
        if ath is None or ath <= 0:
            continue
        pct_ath = pct_from_high(tech.price, ath)
        if (
            cfg.athdip_require_near_ath_pct
            and pct_ath < cfg.athdip_max_pct_from_ath
        ):
            continue
        prior_high: float | None = None
        days_since: int | None = None
        setup_ok = True
        if cfg.athdip_require_multi_year_break:
            setup_ok, prior_high, ath_setup, days_since = recent_multi_year_ath_setup(
                high,
                close,
                fresh_window=cfg.athdip_fresh_high_days,
                max_days_since_ath=cfg.athdip_max_days_since_ath,
            )
            if not setup_ok:
                continue
            if ath_setup is not None:
                ath = ath_setup
                pct_ath = pct_from_high(tech.price, ath)
        candidates.append(
            (ticker, tech, ath, pct_ath, prior_high, days_since, setup_ok)
        )

    logger.info(
        "Athdip daily prefilter: %d recent multi-year ATH setups "
        "(≤%dd since ATH, %dd break window)",
        len(candidates),
        cfg.athdip_max_days_since_ath,
        cfg.athdip_fresh_high_days,
    )
    if not candidates:
        return pd.DataFrame()

    tickers = [t for t, *_ in candidates]
    bars_4h = download_intraday(
        tickers,
        interval="4h",
        period=cfg.athdip_intraday_period,
        chunk_size=min(40, cfg.price_chunk_size),
    )
    rows: list[ScreenRow] = []
    for ticker, tech, ath, pct_ath, prior_high, days_since, setup_ok in candidates:
        close_4h = series_for(bars_4h, "Close", ticker).dropna()
        if close_4h.empty or len(close_4h) < 20:
            continue
        detail = min_rsi_lookback_detail(
            close_4h,
            lookback_days=cfg.athdip_rsi_lookback_days,
            period=14,
        )
        if detail is None:
            continue
        rsi_4h, trig_day = detail
        trigger_date = str(trig_day)
        trigger_price: float | None = None
        daily = close_by_ticker.get(ticker, pd.Series(dtype=float)).dropna()
        if not daily.empty:
            d_idx = pd.to_datetime(daily.index).tz_localize(None).normalize()
            daily = daily.copy()
            daily.index = d_idx
            td = pd.Timestamp(trig_day).normalize()
            if td in daily.index:
                trigger_price = float(daily.loc[td])
            else:
                prior = daily.index[daily.index <= td]
                if len(prior):
                    trigger_price = float(daily.loc[prior[-1]])
        if trigger_price is None and not close_4h.empty:
            # Fallback: 4h close on trigger session
            for ts, px in close_4h.items():
                t = pd.Timestamp(ts)
                sess = (
                    t.tz_convert("America/New_York").date()
                    if t.tzinfo is not None
                    else t.date()
                )
                if sess == trig_day:
                    trigger_price = float(px)
                    break
        pct_ema: float | None = None
        if cfg.athdip_use_4h_ema200 and len(close_4h) >= 200:
            ema200 = close_4h.ewm(span=200, adjust=False).mean()
            ema_last = float(ema200.iloc[-1])
            px_last = float(close_4h.iloc[-1])
            if ema_last > 0:
                pct_ema = (px_last / ema_last - 1.0) * 100.0
        rows.append(
            build_athdip_row(
                ticker,
                tech,
                market,
                ath_high=ath,
                pct_from_ath=pct_ath,
                rsi_4h=rsi_4h,
                config=cfg,
                pct_from_ema200_4h=pct_ema,
                prior_high=prior_high,
                days_since_ath=days_since,
                setup_ok=setup_ok,
                trigger_date=trigger_date,
                trigger_price=trigger_price,
            )
        )
    logger.info(
        "Athdip 4h RSI≤%.0f (min over %dd): %d / %d candidates passed%s",
        cfg.athdip_rsi_4h_max,
        cfg.athdip_rsi_lookback_days,
        sum(1 for r in rows if r.pass_filters),
        len(rows),
        (
            f" · lowest 4h RSI={min(r.rsi_4h for r in rows):.1f}"
            if rows
            else ""
        ),
    )
    return _rows_to_passed(rows)
