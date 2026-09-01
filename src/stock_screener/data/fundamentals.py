from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class FundamentalsSnapshot:
    ticker: str
    sector: str
    industry: str
    # Growth
    revenue_growth: float | None
    earnings_growth: float | None
    # Profitability
    gross_margin: float | None
    operating_margin: float | None
    profit_margin: float | None
    fcf: float | None
    # Balance sheet
    debt_to_equity: float | None
    current_ratio: float | None
    # Capital allocation / returns
    roe: float | None
    roa: float | None
    dividend_yield: float | None
    payout_ratio: float | None
    # Valuation
    trailing_pe: float | None
    forward_pe: float | None
    price_to_sales: float | None
    price_to_book: float | None
    ev_ebitda: float | None
    # Pillar scores 0–100
    growth_score: float
    profitability_score: float
    balance_score: float
    capital_score: float
    valuation_score: float
    composite: float
    pass_filters: bool
    filter_reason: str
    why: str

    def to_dict(self) -> dict:
        return asdict(self)


def _f(info: dict, *keys: str) -> float | None:
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        try:
            x = float(v)
            if x != x:  # NaN
                continue
            return x
        except (TypeError, ValueError):
            continue
    return None


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _score_growth(rev: float | None, eps: float | None) -> float:
    parts: list[float] = []
    for g in (rev, eps):
        if g is None:
            continue
        # 0% → 40, 10% → 70, 25%+ → 95; negative growth penalized
        parts.append(_clip(40.0 + g * 200.0))
    return sum(parts) / len(parts) if parts else 50.0


def _score_profitability(
    gross: float | None,
    op: float | None,
    profit: float | None,
    fcf: float | None,
) -> float:
    parts: list[float] = []
    for m in (gross, op, profit):
        if m is None:
            continue
        parts.append(_clip(30.0 + m * 120.0))  # 20% margin ~ 54, 40% ~ 78
    if fcf is not None:
        parts.append(70.0 if fcf > 0 else 25.0)
    return sum(parts) / len(parts) if parts else 50.0


def _score_balance(de: float | None, current: float | None) -> float:
    parts: list[float] = []
    if de is not None:
        # debt/equity often reported as percent-like (e.g. 50 = 0.5) or ratio
        ratio = de / 100.0 if de > 5 else de
        if ratio <= 0.5:
            parts.append(85.0)
        elif ratio <= 1.0:
            parts.append(70.0)
        elif ratio <= 2.0:
            parts.append(50.0)
        else:
            parts.append(25.0)
    if current is not None:
        if current >= 1.5:
            parts.append(80.0)
        elif current >= 1.0:
            parts.append(60.0)
        else:
            parts.append(30.0)
    return sum(parts) / len(parts) if parts else 50.0


def _score_capital(roe: float | None, roa: float | None, div: float | None) -> float:
    parts: list[float] = []
    for r in (roe, roa):
        if r is None:
            continue
        parts.append(_clip(40.0 + r * 150.0))
    if div is not None and div > 0:
        # modest dividend is fine; very high may be value trap — soft bump
        parts.append(_clip(55.0 + min(div, 0.05) * 400.0))
    return sum(parts) / len(parts) if parts else 50.0


def _score_valuation(
    pe: float | None,
    fwd_pe: float | None,
    ps: float | None,
    pb: float | None,
    ev: float | None,
) -> float:
    """Higher score = cheaper / more reasonable multiple (long-only value bias)."""
    parts: list[float] = []

    def pe_score(x: float) -> float:
        if x <= 0:
            return 20.0
        if x < 12:
            return 90.0
        if x < 20:
            return 75.0
        if x < 30:
            return 55.0
        if x < 45:
            return 40.0
        return 25.0

    for x in (pe, fwd_pe):
        if x is not None:
            parts.append(pe_score(x))
    if ps is not None:
        parts.append(85.0 if ps < 2 else 70.0 if ps < 5 else 50.0 if ps < 10 else 30.0)
    if pb is not None and pb > 0:
        parts.append(80.0 if pb < 2 else 60.0 if pb < 5 else 40.0 if pb < 10 else 25.0)
    if ev is not None and ev > 0:
        parts.append(85.0 if ev < 10 else 65.0 if ev < 15 else 45.0 if ev < 25 else 30.0)
    return sum(parts) / len(parts) if parts else 50.0


def fetch_fundamentals(ticker: str) -> FundamentalsSnapshot | None:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fundamentals fetch failed for %s: %s", ticker, exc)
        return None
    if not info or info.get("trailingPegRatio") == "None" and not info.get("sector"):
        # still try — many fields may exist
        pass

    rev = _f(info, "revenueGrowth")
    eps_g = _f(info, "earningsGrowth", "earningsQuarterlyGrowth")
    gross = _f(info, "grossMargins")
    op = _f(info, "operatingMargins")
    profit = _f(info, "profitMargins")
    fcf = _f(info, "freeCashflow")
    de = _f(info, "debtToEquity")
    current = _f(info, "currentRatio")
    roe = _f(info, "returnOnEquity")
    roa = _f(info, "returnOnAssets")
    div = _f(info, "dividendYield")
    payout = _f(info, "payoutRatio")
    pe = _f(info, "trailingPE")
    fwd = _f(info, "forwardPE")
    ps = _f(info, "priceToSalesTrailing12Months")
    pb = _f(info, "priceToBook")
    ev = _f(info, "enterpriseToEbitda")

    growth = _score_growth(rev, eps_g)
    profitability = _score_profitability(gross, op, profit, fcf)
    balance = _score_balance(de, current)
    capital = _score_capital(roe, roa, div)
    valuation = _score_valuation(pe, fwd, ps, pb, ev)

    composite = round(
        0.20 * growth
        + 0.25 * profitability
        + 0.20 * balance
        + 0.15 * capital
        + 0.20 * valuation,
        2,
    )

    # Soft quality gate: need at least some real fields
    known = sum(
        1
        for x in (rev, eps_g, gross, op, profit, de, current, roe, pe, ps)
        if x is not None
    )
    if known < 3:
        return FundamentalsSnapshot(
            ticker=ticker,
            sector=str(info.get("sector") or ""),
            industry=str(info.get("industry") or ""),
            revenue_growth=rev,
            earnings_growth=eps_g,
            gross_margin=gross,
            operating_margin=op,
            profit_margin=profit,
            fcf=fcf,
            debt_to_equity=de,
            current_ratio=current,
            roe=roe,
            roa=roa,
            dividend_yield=div,
            payout_ratio=payout,
            trailing_pe=pe,
            forward_pe=fwd,
            price_to_sales=ps,
            price_to_book=pb,
            ev_ebitda=ev,
            growth_score=growth,
            profitability_score=profitability,
            balance_score=balance,
            capital_score=capital,
            valuation_score=valuation,
            composite=float("nan"),
            pass_filters=False,
            filter_reason="insufficient fundamentals",
            why="insufficient fundamentals",
        )

    ok = True
    reason = ""
    # Hard-ish quality floors
    if profit is not None and profit < -0.05 and (fcf is None or fcf < 0):
        ok = False
        reason = "unprofitable without FCF"
    elif de is not None:
        ratio = de / 100.0 if de > 5 else de
        if ratio > 3.5:
            ok = False
            reason = "excessive leverage"

    bits = [
        f"g={growth:.0f}",
        f"p={profitability:.0f}",
        f"bs={balance:.0f}",
        f"cap={capital:.0f}",
        f"val={valuation:.0f}",
    ]
    if info.get("sector"):
        bits.insert(0, str(info.get("sector")))

    return FundamentalsSnapshot(
        ticker=ticker,
        sector=str(info.get("sector") or ""),
        industry=str(info.get("industry") or ""),
        revenue_growth=rev,
        earnings_growth=eps_g,
        gross_margin=gross,
        operating_margin=op,
        profit_margin=profit,
        fcf=fcf,
        debt_to_equity=de,
        current_ratio=current,
        roe=roe,
        roa=roa,
        dividend_yield=div,
        payout_ratio=payout,
        trailing_pe=pe,
        forward_pe=fwd,
        price_to_sales=ps,
        price_to_book=pb,
        ev_ebitda=ev,
        growth_score=round(growth, 1),
        profitability_score=round(profitability, 1),
        balance_score=round(balance, 1),
        capital_score=round(capital, 1),
        valuation_score=round(valuation, 1),
        composite=composite if ok else float("nan"),
        pass_filters=ok,
        filter_reason=reason,
        why="; ".join(bits) if ok else reason,
    )


def fetch_fundamentals_map(
    tickers: list[str],
    max_workers: int = 6,
) -> dict[str, FundamentalsSnapshot]:
    out: dict[str, FundamentalsSnapshot] = {}
    if not tickers:
        return out
    logger.info("Fetching fundamentals for %d tickers via yfinance", len(tickers))

    def _one(t: str) -> tuple[str, FundamentalsSnapshot | None]:
        return t, fetch_fundamentals(t)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, snap = fut.result()
            if snap is not None:
                out[t] = snap
    return out
