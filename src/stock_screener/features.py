from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TechnicalFeatures:
    price: float
    support: float
    resistance: float
    proximity: float  # 0 near support, 1 near resistance
    range_pct: float
    sma_fast: float
    sma_slow: float
    above_sma_fast: bool
    above_sma_slow: bool
    golden_cross: bool
    death_cross: bool
    avg_dollar_volume: float
    sma_score: float  # 0–100 for support/long dip bias
    proximity_score: float  # 0–100 for support bias
    # Momentum / catalyst fields
    day_return_pct: float
    volume_ratio: float
    breakout: bool
    momentum_score: float  # 0–100 for catalyst bias
    # Oscillator / range context
    rsi14: float
    rsi_slope: float  # RSI change over last 3 sessions (negative → oversold)
    rsi_bias: str  # oversold / toward oversold / neutral / toward overbought / overbought
    rsi_score_support: float  # higher when oversold (good for dips)
    rsi_score_catalyst: float  # higher when firm momentum, not extreme OB
    high_52w: float
    low_52w: float
    pct_from_52w_high: float  # negative when below high
    pct_from_52w_low: float  # positive when above low
    high_1w: float
    low_1w: float
    pct_from_1w_high: float
    pct_from_1w_low: float


def _last(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI series."""
    c = close.dropna()
    if len(c) < period + 2:
        return pd.Series(dtype=float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss > 1e-12, np.where(avg_gain > 1e-12, 100.0, 50.0))
    return rsi.clip(0.0, 100.0)


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Latest RSI value; returns 50.0 if insufficient history."""
    series = compute_rsi_series(close, period=period)
    if series.empty:
        return 50.0
    return float(series.iloc[-1])


def min_rsi_in_lookback(
    close: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
    lookback_days: int = 5,
    period: int = 14,
) -> float | None:
    """Minimum RSI over bars whose session date falls in ``[as_of-lookback, as_of]``."""
    detail = min_rsi_lookback_detail(
        close, as_of=as_of, lookback_days=lookback_days, period=period
    )
    return None if detail is None else detail[0]


def min_rsi_lookback_detail(
    close: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
    lookback_days: int = 5,
    period: int = 14,
) -> tuple[float, object] | None:
    """Return ``(min_rsi, trigger_session_date)`` in the lookback window.

    ``trigger_session_date`` is the calendar date of the 4h bar that printed the
    minimum RSI (earliest bar on ties).
    """
    c = close.dropna()
    if c.empty or lookback_days < 0:
        return None
    rsi = compute_rsi_series(c, period=period)
    if rsi.empty or len(c) < period + 2:
        return None

    def _session_date(ts) -> object:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            return t.tz_convert("America/New_York").date()
        return t.date()

    if as_of is None:
        end_date = _session_date(rsi.index[-1])
    else:
        a = pd.Timestamp(as_of)
        end_date = a.tz_convert("America/New_York").date() if a.tzinfo else a.date()
    start_date = pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)
    start_d = start_date.date()

    best: tuple[float, object, object] | None = None  # rsi, date, ts
    for ts, val in rsi.items():
        if not pd.notna(val):
            continue
        d = _session_date(ts)
        if not (start_d <= d <= end_date):
            continue
        v = float(val)
        if best is None or v < best[0] or (v == best[0] and ts < best[2]):
            best = (v, d, ts)
    if best is None:
        return None
    return best[0], best[1]

def all_time_high(high: pd.Series) -> float | None:
    """Max high over available history (ATH proxy for downloaded window)."""
    h = high.dropna()
    if h.empty:
        return None
    return float(h.max())


def pct_from_high(price: float, high: float) -> float:
    if high <= 0:
        return 0.0
    return ((price / high) - 1.0) * 100.0


def multi_year_ath_breakout(
    high: pd.Series,
    close: pd.Series,
    *,
    window: int = 21,
) -> tuple[bool, float | None, float | None, int | None]:
    """Fresh multi-year ATH breakout check (same-bar setup).

    Gates:
      • close > prior_high (max high excluding last ``window`` sessions)
      • recent_max > prior_high (a new high was logged inside that window)

    Returns ``(ok, prior_high, recent_max, days_since_ath)``.
    ``days_since_ath`` is trading sessions since the absolute high in the series.
    """
    if window < 1:
        return False, None, None, None
    h = high.dropna()
    c = close.dropna()
    if h.empty or c.empty:
        return False, None, None, None
    idx = h.index.intersection(c.index)
    if len(idx) < window + 2:
        return False, None, None, None
    h = h.loc[idx]
    c = c.loc[idx]
    prior = h.iloc[:-window]
    recent = h.iloc[-window:]
    if prior.empty or recent.empty:
        return False, None, None, None
    prior_high = float(prior.max())
    recent_max = float(recent.max())
    if prior_high <= 0:
        return False, None, None, None
    price = float(c.iloc[-1])
    ath_pos = int(h.to_numpy().argmax())
    days_since_ath = len(h) - 1 - ath_pos
    ok = price > prior_high and recent_max > prior_high
    return ok, prior_high, recent_max, days_since_ath


def recent_multi_year_ath_setup(
    high: pd.Series,
    close: pd.Series,
    *,
    fresh_window: int = 21,
    max_days_since_ath: int = 63,
) -> tuple[bool, float | None, float | None, int | None]:
    """Setup for athdip: multi-year ATH was printed recently (then wait for RSI dip).

    Unlike ``multi_year_ath_breakout``, this does **not** require price still above
    the prior high — only that the absolute high in the series:
      • occurred within ``max_days_since_ath`` trading sessions, and
      • was a multi-year break at the time it printed (fresh_window logic on the
        slice ending at the ATH bar).

    Returns ``(ok, prior_high_at_ath, ath_high, days_since_ath)``.
    """
    if fresh_window < 1 or max_days_since_ath < 0:
        return False, None, None, None
    h = high.dropna()
    c = close.dropna()
    if h.empty or c.empty:
        return False, None, None, None
    idx = h.index.intersection(c.index)
    if len(idx) < fresh_window + 2:
        return False, None, None, None
    h = h.loc[idx]
    c = c.loc[idx]
    ath_pos = int(h.to_numpy().argmax())
    days_since_ath = len(h) - 1 - ath_pos
    ath_high = float(h.iloc[ath_pos])
    if days_since_ath > max_days_since_ath:
        return False, None, ath_high, days_since_ath
    # Evaluate breakout on history ending at the ATH bar (inclusive)
    h_at = h.iloc[: ath_pos + 1]
    c_at = c.iloc[: ath_pos + 1]
    ok_break, prior_high, _, _ = multi_year_ath_breakout(
        h_at, c_at, window=fresh_window
    )
    if not ok_break:
        # Short listings: treat series ATH as setup if we have any prior range
        if prior_high is None and len(h_at) >= fresh_window + 2:
            prior_high = float(h_at.iloc[:-fresh_window].max())
            ok_break = ath_high > prior_high
        else:
            return False, prior_high, ath_high, days_since_ath
    return True, prior_high, ath_high, days_since_ath


def rsi_bias_label(rsi: float, slope: float, *, slope_threshold: float = 2.0) -> str:
    """Classify RSI level and short-term bend toward oversold or overbought."""
    if rsi <= 30:
        return "oversold"
    if rsi >= 70:
        return "overbought"
    if slope <= -slope_threshold:
        return "toward oversold"
    if slope >= slope_threshold:
        return "toward overbought"
    return "neutral"


def _rsi_support_score(rsi: float) -> float:
    # Prefer oversold / pullback zone for mean-reversion longs
    if rsi <= 30:
        return 90.0
    if rsi <= 40:
        return 80.0
    if rsi <= 50:
        return 65.0
    if rsi <= 60:
        return 45.0
    if rsi <= 70:
        return 30.0
    return 15.0


def _rsi_catalyst_score(rsi: float) -> float:
    # Prefer firm momentum; penalize washed-out and extreme overbought chase
    if 55 <= rsi <= 70:
        return 85.0
    if 45 <= rsi < 55:
        return 65.0
    if 70 < rsi <= 80:
        return 55.0
    if rsi > 80:
        return 35.0
    if 35 <= rsi < 45:
        return 45.0
    return 25.0


def compute_technicals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    *,
    donchian_window: int = 55,
    sma_fast: int = 50,
    sma_slow: int = 200,
    avg_volume_window: int = 20,
) -> TechnicalFeatures | None:
    if close.dropna().shape[0] < sma_slow + 5:
        return None

    price = _last(close)
    if price is None or price <= 0:
        return None

    roll_high = high.shift(1).rolling(donchian_window).max()
    roll_low = low.shift(1).rolling(donchian_window).min()
    resistance = _last(roll_high)
    support = _last(roll_low)
    if resistance is None or support is None:
        return None
    if resistance <= support:
        return None

    proximity = (price - support) / (resistance - support)
    proximity = float(np.clip(proximity, 0.0, 1.0))
    mid = (resistance + support) / 2.0
    range_pct = (resistance - support) / mid if mid else 0.0

    sma_f = close.rolling(sma_fast).mean()
    sma_s = close.rolling(sma_slow).mean()
    sma_fast_v = _last(sma_f)
    sma_slow_v = _last(sma_s)
    if sma_fast_v is None or sma_slow_v is None:
        return None

    if sma_f.dropna().shape[0] < 2 or sma_s.dropna().shape[0] < 2:
        golden = death = False
    else:
        prev_fast, prev_slow = float(sma_f.dropna().iloc[-2]), float(sma_s.dropna().iloc[-2])
        golden = prev_fast <= prev_slow and sma_fast_v > sma_slow_v
        death = prev_fast >= prev_slow and sma_fast_v < sma_slow_v

    dollar_vol = (close * volume).rolling(avg_volume_window).mean()
    adv = _last(dollar_vol) or 0.0

    above_fast = price >= sma_fast_v
    above_slow = price >= sma_slow_v

    if not above_slow:
        sma_score = 20.0 if sma_fast_v >= sma_slow_v else 5.0
    elif above_fast:
        dist = (price - sma_fast_v) / sma_fast_v
        sma_score = 70.0 - min(30.0, dist * 200.0)
    else:
        sma_score = 90.0

    if sma_fast_v > sma_slow_v:
        sma_score = min(100.0, sma_score + 5.0)
    else:
        sma_score = max(0.0, sma_score - 15.0)

    proximity_score = (1.0 - proximity) * 100.0

    c = close.dropna()
    h = high.reindex(c.index).ffill()
    l = low.reindex(c.index).ffill()
    v = volume.reindex(c.index).fillna(0.0)
    if len(c) >= 2:
        day_return_pct = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
    else:
        day_return_pct = 0.0
    vol_ma = v.rolling(avg_volume_window).mean()
    vol_last = float(v.iloc[-1]) if len(v) else 0.0
    vol_avg = float(vol_ma.dropna().iloc[-1]) if not vol_ma.dropna().empty else 0.0
    volume_ratio = (vol_last / vol_avg) if vol_avg > 0 else 1.0

    breakout = price >= resistance * 0.998

    rsi_series = compute_rsi_series(c, period=14)
    rsi14 = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    rsi_lookback = 3
    if len(rsi_series) > rsi_lookback:
        rsi_slope = float(rsi_series.iloc[-1] - rsi_series.iloc[-1 - rsi_lookback])
    else:
        rsi_slope = 0.0
    rsi_bias = rsi_bias_label(rsi14, rsi_slope)
    rsi_support = _rsi_support_score(rsi14)
    rsi_catalyst = _rsi_catalyst_score(rsi14)

    # 52-week (~252 trading days) and 1-week (5 sessions) highs/lows
    win_52 = min(252, len(h))
    high_52w = float(h.tail(win_52).max()) if win_52 else price
    low_52w = float(l.tail(win_52).min()) if win_52 else price
    win_1w = min(5, len(h))
    high_1w = float(h.tail(win_1w).max()) if win_1w else price
    low_1w = float(l.tail(win_1w).min()) if win_1w else price

    pct_from_52w_high = (
        ((price / high_52w) - 1.0) * 100.0 if high_52w > 0 else 0.0
    )
    pct_from_52w_low = ((price / low_52w) - 1.0) * 100.0 if low_52w > 0 else 0.0
    pct_from_1w_high = ((price / high_1w) - 1.0) * 100.0 if high_1w > 0 else 0.0
    pct_from_1w_low = ((price / low_1w) - 1.0) * 100.0 if low_1w > 0 else 0.0

    # Catalyst momentum: upside + volume + breakout + RSI + near 52w high
    mom = 50.0
    mom += max(-20.0, min(25.0, day_return_pct * 4.0))
    if volume_ratio >= 2.0:
        mom += 20.0
    elif volume_ratio >= 1.4:
        mom += 12.0
    elif volume_ratio < 0.8:
        mom -= 10.0
    if breakout:
        mom += 15.0
    elif proximity >= 0.7:
        mom += 8.0
    if above_fast and not above_slow:
        mom += 5.0
    if above_fast and above_slow:
        mom += 8.0
    mom += (rsi_catalyst - 50.0) * 0.25
    if pct_from_52w_high >= -3.0:
        mom += 10.0  # pressing / at 52w high
    elif pct_from_52w_high <= -25.0:
        mom -= 8.0
    momentum_score = float(np.clip(mom, 0.0, 100.0))

    # Support: bump sma/prox context when RSI oversold and near 52w/week lows
    if rsi14 <= 40 and pct_from_52w_low <= 15.0:
        sma_score = min(100.0, sma_score + 5.0)
    if rsi14 >= 70:
        sma_score = max(0.0, sma_score - 8.0)

    return TechnicalFeatures(
        price=price,
        support=support,
        resistance=resistance,
        proximity=proximity,
        range_pct=range_pct,
        sma_fast=sma_fast_v,
        sma_slow=sma_slow_v,
        above_sma_fast=above_fast,
        above_sma_slow=above_slow,
        golden_cross=golden,
        death_cross=death,
        avg_dollar_volume=adv,
        sma_score=float(np.clip(sma_score, 0.0, 100.0)),
        proximity_score=float(np.clip(proximity_score, 0.0, 100.0)),
        day_return_pct=round(day_return_pct, 3),
        volume_ratio=round(volume_ratio, 3),
        breakout=breakout,
        momentum_score=momentum_score,
        rsi14=round(rsi14, 2),
        rsi_slope=round(rsi_slope, 2),
        rsi_bias=rsi_bias,
        rsi_score_support=round(rsi_support, 1),
        rsi_score_catalyst=round(rsi_catalyst, 1),
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        pct_from_52w_high=round(pct_from_52w_high, 2),
        pct_from_52w_low=round(pct_from_52w_low, 2),
        high_1w=round(high_1w, 2),
        low_1w=round(low_1w, 2),
        pct_from_1w_high=round(pct_from_1w_high, 2),
        pct_from_1w_low=round(pct_from_1w_low, 2),
    )
