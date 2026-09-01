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


# Back-compat alias
def build_row(*args, **kwargs) -> ScreenRow:
    return build_support_row(*args, **kwargs)
