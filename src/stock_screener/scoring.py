from __future__ import annotations

from dataclasses import asdict, dataclass

from stock_screener.config import ScreenerConfig
from stock_screener.data.earnings import EarningsAssessment
from stock_screener.data.grok_x import GrokXAssessment
from stock_screener.data.news import NewsAssessment
from stock_screener.data.reddit import RedditAssessment
from stock_screener.data.sentiment import MarketSentiment
from stock_screener.features import TechnicalFeatures


@dataclass
class ScreenRow:
    mode: str
    ticker: str
    price: float
    support: float
    resistance: float
    proximity: float
    sma50: float
    sma200: float
    sma_regime: str
    rsi14: float
    rsi_slope: float
    rsi_bias: str
    high_52w: float
    low_52w: float
    pct_from_52w_high: float
    pct_from_52w_low: float
    high_1w: float
    low_1w: float
    pct_from_1w_high: float
    pct_from_1w_low: float
    day_return_pct: float
    volume_ratio: float
    breakout: bool
    momentum_score: float
    market_label: str
    news_score: float
    news_headline: str
    days_to_earnings: int | None
    earnings_score: float
    reddit_score: float
    reddit_mentions: int
    grok_score: float
    grok_label: str
    social_score: float
    composite: float
    pass_filters: bool
    filter_reason: str
    why: str
    # Athdip extras (defaults keep other modes unchanged)
    rsi_4h: float = 50.0
    ath_high: float = 0.0
    pct_from_ath: float = 0.0
    prior_high: float = 0.0
    days_since_ath: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


def sma_regime_label(tech: TechnicalFeatures) -> str:
    if tech.death_cross:
        return "death-cross"
    if tech.golden_cross:
        return "golden-cross"
    if tech.above_sma_slow and tech.above_sma_fast:
        return "above-both"
    if tech.above_sma_slow and not tech.above_sma_fast:
        return "pullback-to-50"
    if not tech.above_sma_slow and tech.above_sma_fast:
        return "above-50-only"
    return "below-both"


def blend_social(
    reddit: RedditAssessment,
    grok: GrokXAssessment | None,
) -> tuple[float, str]:
    if grok is not None and grok.used and grok.label not in {"error", "skipped"}:
        score = 0.45 * reddit.score + 0.55 * grok.score
        detail = f"reddit={reddit.score:.0f}; grok={grok.label}"
    else:
        score = reddit.score
        detail = reddit.reason
    return score, detail


def apply_support_filters(
    tech: TechnicalFeatures,
    news: NewsAssessment,
    config: ScreenerConfig,
    market: MarketSentiment,
) -> tuple[bool, str]:
    regime = market.regime
    if tech.price < config.min_price:
        return False, "price too low"
    if tech.avg_dollar_volume < config.min_avg_dollar_volume:
        return False, "illiquid"
    if tech.range_pct < config.min_range_pct:
        return False, "range too tight"
    if tech.proximity > regime.max_proximity:
        return False, "too close to resistance"
    if regime.require_above_sma_slow and not tech.above_sma_slow:
        return False, "below SMA200"
    if config.drop_death_cross and tech.death_cross:
        return False, "death cross"
    if news.veto:
        return False, "news veto"
    return True, ""


def apply_catalyst_filters(
    tech: TechnicalFeatures,
    news: NewsAssessment,
    earnings: EarningsAssessment,
    config: ScreenerConfig,
    market: MarketSentiment,
) -> tuple[bool, str]:
    regime = market.regime
    if tech.price < config.min_price:
        return False, "price too low"
    if tech.avg_dollar_volume < config.min_avg_dollar_volume:
        return False, "illiquid"
    if news.veto:
        return False, "news veto"

    news_signal = news.score >= regime.catalyst_min_news
    volume_signal = tech.volume_ratio >= regime.catalyst_min_volume
    price_signal = tech.day_return_pct >= regime.catalyst_min_day_return
    earnings_signal = earnings.catalyst_score >= 70.0
    breakout_signal = tech.breakout

    if config.catalyst_require_signal and not (
        news_signal or volume_signal or earnings_signal or breakout_signal or price_signal
    ):
        return False, "no catalyst signal"

    if regime.catalyst_require_price_and_volume and not (volume_signal and price_signal):
        return False, "risk-off needs vol+green day"

    if news.score < 55 and tech.day_return_pct < 0 and tech.volume_ratio < 1.1:
        return False, "no upside confirmation"

    return True, ""


def support_composite(
    tech: TechnicalFeatures,
    market: MarketSentiment,
    news: NewsAssessment,
    earnings: EarningsAssessment,
    social_score: float,
    config: ScreenerConfig,
) -> float:
    w_mkt = config.weight_market + market.regime.support_market_weight_boost
    w_news = max(0.05, config.weight_news - market.regime.support_market_weight_boost)
    score = (
        config.weight_proximity * tech.proximity_score
        + config.weight_sma * tech.sma_score
        + config.weight_rsi * tech.rsi_score_support
        + w_mkt * market.score
        + w_news * news.score
        + config.weight_earnings * earnings.score
        + config.weight_social * social_score
    )
    # Soft nudge: near 52w low helps mean-reversion
    if tech.pct_from_52w_low <= 12.0 and tech.rsi14 <= config.support_prefer_rsi_below:
        score = min(100.0, score + 2.0)
    return round(float(score), 2)


def catalyst_composite(
    tech: TechnicalFeatures,
    market: MarketSentiment,
    news: NewsAssessment,
    earnings: EarningsAssessment,
    social_score: float,
    config: ScreenerConfig,
) -> float:
    boost = market.regime.catalyst_momentum_boost
    w_mom = max(0.10, config.cat_weight_momentum + boost)
    w_news = max(0.15, config.cat_weight_news - boost)
    score = (
        w_news * news.score
        + w_mom * tech.momentum_score
        + config.cat_weight_rsi * tech.rsi_score_catalyst
        + config.cat_weight_earnings * earnings.catalyst_score
        + config.cat_weight_social * social_score
        + config.cat_weight_market * market.score
    )
    if tech.rsi14 >= config.catalyst_rsi_overbought:
        score = max(0.0, score - 4.0)
    elif tech.rsi14 >= config.catalyst_prefer_rsi_above and tech.pct_from_52w_high >= -5.0:
        score = min(100.0, score + 2.0)
    return round(float(score), 2)


def _base_row(
    mode: str,
    ticker: str,
    tech: TechnicalFeatures,
    market: MarketSentiment,
    news: NewsAssessment,
    earnings: EarningsAssessment,
    reddit: RedditAssessment,
    grok: GrokXAssessment | None,
    social_score: float,
    composite: float,
    ok: bool,
    reason: str,
    why: str,
) -> ScreenRow:
    return ScreenRow(
        mode=mode,
        ticker=ticker,
        price=round(tech.price, 2),
        support=round(tech.support, 2),
        resistance=round(tech.resistance, 2),
        proximity=round(tech.proximity, 3),
        sma50=round(tech.sma_fast, 2),
        sma200=round(tech.sma_slow, 2),
        sma_regime=sma_regime_label(tech),
        rsi14=tech.rsi14,
        rsi_slope=tech.rsi_slope,
        rsi_bias=tech.rsi_bias,
        high_52w=tech.high_52w,
        low_52w=tech.low_52w,
        pct_from_52w_high=tech.pct_from_52w_high,
        pct_from_52w_low=tech.pct_from_52w_low,
        high_1w=tech.high_1w,
        low_1w=tech.low_1w,
        pct_from_1w_high=tech.pct_from_1w_high,
        pct_from_1w_low=tech.pct_from_1w_low,
        day_return_pct=tech.day_return_pct,
        volume_ratio=tech.volume_ratio,
        breakout=tech.breakout,
        momentum_score=round(tech.momentum_score, 1),
        market_label=market.label,
        news_score=round(news.score, 1),
        news_headline=news.top_headline,
        days_to_earnings=earnings.days_to_earnings,
        earnings_score=round(
            earnings.score if mode == "support" else earnings.catalyst_score, 1
        ),
        reddit_score=round(reddit.score, 1),
        reddit_mentions=reddit.mentions,
        grok_score=round(grok.score, 1) if grok and grok.used else 50.0,
        grok_label=grok.label if grok and grok.used else "skipped",
        social_score=round(social_score, 1),
        composite=composite,
        pass_filters=ok,
        filter_reason=reason,
        why=why if ok else reason,
    )


def build_support_row(
    ticker: str,
    tech: TechnicalFeatures,
    market: MarketSentiment,
    news: NewsAssessment,
    earnings: EarningsAssessment,
    reddit: RedditAssessment,
    grok: GrokXAssessment | None,
    config: ScreenerConfig,
) -> ScreenRow:
    ok, reason = apply_support_filters(tech, news, config, market)
    social_score, social_detail = blend_social(reddit, grok)
    score = (
        support_composite(tech, market, news, earnings, social_score, config)
        if ok
        else float("nan")
    )
    why = "; ".join(
        [
            f"prox={tech.proximity:.2f}",
            f"RSI={tech.rsi14:.0f} ({tech.rsi_bias})",
            f"52w={tech.pct_from_52w_high:+.1f}%H/{tech.pct_from_52w_low:+.1f}%L",
            sma_regime_label(tech),
            market.label,
            news.reason,
            earnings.reason,
            social_detail,
        ]
    )
    return _base_row(
        "support",
        ticker,
        tech,
        market,
        news,
        earnings,
        reddit,
        grok,
        social_score,
        score,
        ok,
        reason,
        why,
    )


def build_catalyst_row(
    ticker: str,
    tech: TechnicalFeatures,
    market: MarketSentiment,
    news: NewsAssessment,
    earnings: EarningsAssessment,
    reddit: RedditAssessment,
    grok: GrokXAssessment | None,
    config: ScreenerConfig,
) -> ScreenRow:
    ok, reason = apply_catalyst_filters(tech, news, earnings, config, market)
    social_score, social_detail = blend_social(reddit, grok)
    score = (
        catalyst_composite(tech, market, news, earnings, social_score, config)
        if ok
        else float("nan")
    )
    flags = []
    if tech.breakout:
        flags.append("breakout")
    if tech.volume_ratio >= market.regime.catalyst_min_volume:
        flags.append(f"vol×{tech.volume_ratio:.1f}")
    if tech.day_return_pct >= market.regime.catalyst_min_day_return:
        flags.append(f"day{tech.day_return_pct:+.1f}%")
    if tech.rsi14 >= 55 or tech.rsi_bias in {"toward overbought", "overbought"}:
        flags.append(f"RSI={tech.rsi14:.0f} ({tech.rsi_bias})")
    if tech.pct_from_52w_high >= -5:
        flags.append("near52wH")
    why = "; ".join(
        [
            " ".join(flags) if flags else "catalyst",
            news.reason,
            earnings.catalyst_reason,
            f"mom={tech.momentum_score:.0f}",
            f"52w={tech.pct_from_52w_high:+.1f}%H",
            market.label,
            social_detail,
        ]
    )
    return _base_row(
        "catalyst",
        ticker,
        tech,
        market,
        news,
        earnings,
        reddit,
        grok,
        social_score,
        score,
        ok,
        reason,
        why,
    )


def apply_athdip_filters(
    tech: TechnicalFeatures,
    *,
    pct_from_ath: float,
    rsi_4h: float,
    config: ScreenerConfig,
    pct_from_ema200_4h: float | None = None,
    prior_high: float | None = None,
    recent_max: float | None = None,
    days_since_ath: int | None = None,
    setup_ok: bool | None = None,
) -> tuple[bool, str]:
    """Sequential athdip: recent multi-year ATH setup, then 4h RSI ≤ max."""
    if tech.price < config.min_price:
        return False, "price below min"
    if tech.avg_dollar_volume < config.min_avg_dollar_volume:
        return False, "illiquid"
    if config.athdip_require_uptrend:
        if config.athdip_require_sma50:
            if not (tech.above_sma_fast and tech.above_sma_slow):
                return False, "not uptrend (need above SMA50 & SMA200)"
        elif not tech.above_sma_slow:
            return False, "not uptrend (need above SMA200)"
    if config.athdip_drop_death_cross and tech.death_cross:
        return False, "death cross"
    if config.athdip_require_multi_year_break:
        if setup_ok is False:
            return False, "no recent multi-year ATH setup"
        if days_since_ath is None:
            return False, "insufficient history for multi-year high"
        if days_since_ath > config.athdip_max_days_since_ath:
            return False, (
                f"ATH too old ({days_since_ath}d > "
                f"{config.athdip_max_days_since_ath}d watch window)"
            )
    if (
        config.athdip_require_near_ath_pct
        and pct_from_ath < config.athdip_max_pct_from_ath
    ):
        return False, f"too far from ATH ({pct_from_ath:.1f}% < {config.athdip_max_pct_from_ath:.0f}%)"
    # Trigger: 4h RSI ≤ max (default 30; charts often kiss 30)
    if rsi_4h > config.athdip_rsi_4h_max:
        return False, (
            f"4h RSI {rsi_4h:.1f} not ≤ {config.athdip_rsi_4h_max:.0f}"
        )
    if (
        config.athdip_use_4h_ema200
        and pct_from_ema200_4h is not None
        and pct_from_ema200_4h < config.athdip_ema200_max_pct
    ):
        return False, (
            f"too far below 4h EMA200 ({pct_from_ema200_4h:.1f}% < "
            f"{config.athdip_ema200_max_pct:.0f}%)"
        )
    return True, ""


def athdip_composite(
    tech: TechnicalFeatures,
    *,
    pct_from_ath: float,
    rsi_4h: float,
    config: ScreenerConfig,
    pct_from_ema200_4h: float | None = None,
    days_since_ath: int | None = None,
) -> float:
    max_rsi = config.athdip_rsi_4h_max
    rsi_score = (
        max(0.0, min(100.0, 100.0 * (1.0 - rsi_4h / max_rsi))) if max_rsi > 0 else 50.0
    )
    span = abs(config.athdip_max_pct_from_ath) or 20.0
    ath_score = max(0.0, min(100.0, 100.0 * (1.0 + pct_from_ath / span)))
    # Prefer fresher ATH within the watch window
    age_score = 70.0
    if days_since_ath is not None and config.athdip_max_days_since_ath > 0:
        age_score = max(
            30.0,
            min(
                100.0,
                100.0
                * (1.0 - days_since_ath / float(config.athdip_max_days_since_ath)),
            ),
        )
    return round(0.55 * rsi_score + 0.25 * ath_score + 0.20 * age_score, 2)
def build_athdip_row(
    ticker: str,
    tech: TechnicalFeatures,
    market: MarketSentiment,
    *,
    ath_high: float,
    pct_from_ath: float,
    rsi_4h: float,
    config: ScreenerConfig,
    pct_from_ema200_4h: float | None = None,
    prior_high: float | None = None,
    recent_max: float | None = None,
    days_since_ath: int | None = None,
    setup_ok: bool | None = None,
) -> ScreenRow:
    from stock_screener.data.earnings import EarningsAssessment
    from stock_screener.data.news import NewsAssessment
    from stock_screener.data.reddit import RedditAssessment

    news = NewsAssessment(50.0, False, 0, "", "n/a")
    earn = EarningsAssessment(None, 50.0, "n/a", 50.0, "n/a")
    reddit = RedditAssessment(50.0, 0, "n/a")
    ok, reason = apply_athdip_filters(
        tech,
        pct_from_ath=pct_from_ath,
        rsi_4h=rsi_4h,
        config=config,
        pct_from_ema200_4h=pct_from_ema200_4h,
        prior_high=prior_high,
        recent_max=recent_max,
        days_since_ath=days_since_ath,
        setup_ok=setup_ok,
    )
    score = (
        athdip_composite(
            tech,
            pct_from_ath=pct_from_ath,
            rsi_4h=rsi_4h,
            config=config,
            pct_from_ema200_4h=pct_from_ema200_4h,
            days_since_ath=days_since_ath,
        )
        if ok
        else float("nan")
    )
    ema_bit = (
        f"4hEMA200={pct_from_ema200_4h:+.1f}%; "
        if pct_from_ema200_4h is not None
        else ""
    )
    prior_bit = ""
    if prior_high is not None and prior_high > 0:
        prior_bit = f"priorHigh={prior_high:.2f}; "
    fresh_bit = ""
    if days_since_ath is not None and days_since_ath >= 0:
        fresh_bit = f"ATH={days_since_ath}d ago; "
    why = (
        f"ATH={ath_high:.2f} ({pct_from_ath:+.1f}%); "
        f"{prior_bit}{fresh_bit}"
        f"4h RSI={rsi_4h:.1f}; {ema_bit}"
        f"daily RSI={tech.rsi14:.0f}; "
        f"{sma_regime_label(tech)}; {market.label}"
    )
    row = _base_row(
        "athdip",
        ticker,
        tech,
        market,
        news,
        earn,
        reddit,
        None,
        50.0,
        score,
        ok,
        reason,
        why,
    )
    row.rsi_4h = round(rsi_4h, 2)
    row.ath_high = round(ath_high, 2)
    row.pct_from_ath = round(pct_from_ath, 2)
    row.prior_high = round(prior_high, 2) if prior_high is not None else 0.0
    row.days_since_ath = int(days_since_ath) if days_since_ath is not None else -1
    return row


# Back-compat alias
def build_row(*args, **kwargs) -> ScreenRow:
    return build_support_row(*args, **kwargs)
