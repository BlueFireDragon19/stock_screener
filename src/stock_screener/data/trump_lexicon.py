"""Keyword / industry → ticker map for Trump post & news matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Cashtags like $AAPL
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")

# Prefer full names / phrases; short tokens matched with word boundaries below.
_COMPANY_NAMES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta platforms": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "boeing": "BA",
    "intel": "INTC",
    "micron": "MU",
    "micron technology": "MU",
    "dell": "DELL",
    "dell technologies": "DELL",
    "advanced micro devices": "AMD",
    "exxon": "XOM",
    "chevron": "CVX",
    "goldman": "GS",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "walmart": "WMT",
    "disney": "DIS",
    "netflix": "NFLX",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "lockheed": "LMT",
    "raytheon": "RTX",
    "palantir": "PLTR",
    "coinbase": "COIN",
    "riot platforms": "RIOT",
    "riot blockchain": "RIOT",
    "marathon digital": "MARA",
    "globalfoundries": "GFS",
    "mp materials": "MP",
    "u.s. steel": "X",
    "us steel": "X",
    "ibm": "IBM",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "taiwan semiconductor": "TSM",
    "tsmc": "TSM",
}

# Govt equity / industrial-policy language (CHIPS, Commerce stakes, etc.)
_GOVT_INVEST_KEYWORDS = (
    "equity stake",
    "government stake",
    "federal equity",
    "minority stake",
    "golden share",
    "equity investment",
    "government investment",
    "took a stake",
    "stake in",
    "chips act",
    "chips and science",
    "commerce department",
    "investment fund path",
    "government owns",
    "uncle sam shareholder",
    "equity portfolio",
    "warrant and common stock",
)

# Known / frequent public recipients of admin equity or industrial-policy capital
_GOVT_EQUITY_TICKERS = [
    "INTC",
    "MU",
    "DELL",
    "GFS",
    "MP",
    "X",
    "IBM",
    "LMT",
    "NVDA",
    "AMD",
    "TSM",
    "SMH",
]


# Industry / policy themes → liquid proxies (ETFs + leaders)
INDUSTRY_RULES: list[tuple[str, list[str], list[str]]] = [
    # (theme, keywords, tickers)
    (
        "tariffs_trade",
        [
            "tariff",
            "tariffs",
            "import tax",
            "china trade",
            "trade war",
            "reciprocal trade",
            "trade deal",
        ],
        ["XLI", "CAT", "DE", "BA", "F", "GM", "NKE", "AAPL"],
    ),
    (
        "china",
        ["china", "beijing", "xi jinping", "chinese communist", "chinese"],
        ["FXI", "BABA", "JD", "PDD", "NIO", "AAPL", "QCOM", "NVDA"],
    ),
    (
        "energy_oil",
        ["crude", "opec", "drilling", "petroleum", "lng", "natural gas", "oil prices", "oil"],
        ["XLE", "XOM", "CVX", "COP", "SLB", "OXY"],
    ),
    (
        "energy_green",
        ["electric vehicle", "green new deal", "climate", "solar", "wind power", "evs"],
        ["TSLA", "F", "GM", "ENPH", "FSLR", "NEE", "ICLN"],
    ),
    (
        "defense",
        [
            "defense",
            "military",
            "pentagon",
            "nato",
            "missile",
            "missiles",
            "israel",
            "ukraine",
            "iran war",
            "war in",
        ],
        ["XAR", "LMT", "RTX", "NOC", "GD", "BA"],
    ),
    (
        "pharma_health",
        ["pharma", "drug price", "vaccine", "medicare", "obamacare", "medicaid"],
        ["XLV", "PFE", "MRK", "JNJ", "LLY", "UNH", "ABBV"],
    ),
    (
        "banks_finance",
        [
            "wall street",
            "interest rate",
            "interest rates",
            "powell",
            "crypto",
            "bitcoin",
            "the fed",
            "federal reserve",
            "banks",
        ],
        ["XLF", "JPM", "BAC", "GS", "MS", "COIN", "MSTR", "IBIT"],
    ),
    (
        "semis_tech",
        [
            "semiconductor",
            "semiconductors",
            "artificial intelligence",
            "nvidia",
            "taiwan",
            "chips",
            "chip stocks",
            "chipmakers",
            "chipmaker",
        ],
        ["SMH", "NVDA", "AVGO", "TSM", "AMD", "INTC", "MU", "ASML", "DELL", "GFS"],
    ),
    (
        "govt_equity",
        list(_GOVT_INVEST_KEYWORDS),
        list(_GOVT_EQUITY_TICKERS),
    ),
    (
        "auto",
        ["automobile", "detroit", "uaw", "car company", "auto industry", "automakers"],
        ["F", "GM", "TSLA", "RIVN", "CARZ"],
    ),
    (
        "steel_materials",
        ["steel", "aluminum", "aluminium", "copper", "mining"],
        ["X", "NUE", "CLF", "FCX", "AA", "XLB"],
    ),
    (
        "immigration_labor",
        ["border", "immigration", "deport", "deportation", "migrant", "migrants"],
        ["XLI", "CAT", "DE", "UNP", "CSX"],
    ),
]


def all_tracked_tickers() -> list[str]:
    """Deduped liquid tickers from company + industry maps (backtest / enrich)."""
    out: list[str] = []
    seen: set[str] = set()
    for t in _COMPANY_NAMES.values():
        if t not in seen:
            seen.add(t)
            out.append(t)
    for _theme, _kws, tickers in INDUSTRY_RULES:
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
    for t in ("META", "AMD"):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@dataclass
class MatchHit:
    ticker: str
    theme: str
    keyword: str
    weight: float  # higher = more direct mention


@lru_cache(maxsize=256)
def _phrase_re(phrase: str) -> re.Pattern[str]:
    """Word-boundary match for a phrase (spaces allowed inside)."""
    parts = [re.escape(p) for p in phrase.strip().lower().split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


def _has_phrase(text: str, phrase: str) -> bool:
    return _phrase_re(phrase).search(text) is not None


def extract_matches(text: str) -> list[MatchHit]:
    """Find tickers / industries mentioned in free text."""
    if not text:
        return []
    hits: dict[tuple[str, str], MatchHit] = {}

    for m in _CASHTAG.finditer(text.upper()):
        t = m.group(1)
        key = (t, "cashtag")
        hits[key] = MatchHit(t, "cashtag", f"${t}", 1.0)

    company_tickers: set[str] = set()
    for name, ticker in _COMPANY_NAMES.items():
        if _has_phrase(text, name):
            key = (ticker, "company")
            hits[key] = MatchHit(ticker, "company", name, 0.9)
            company_tickers.add(ticker)

    # Standalone short mega-caps that are too ambiguous as substrings
    for name, ticker in (("meta", "META"), ("amd", "AMD")):
        if _has_phrase(text, name) and (ticker, "company") not in hits:
            hits[(ticker, "company")] = MatchHit(ticker, "company", name, 0.85)
            company_tickers.add(ticker)

    govt_hit = next((kw for kw in _GOVT_INVEST_KEYWORDS if _has_phrase(text, kw)), None)

    for theme, keywords, tickers in INDUSTRY_RULES:
        for kw in keywords:
            if _has_phrase(text, kw):
                for t in tickers:
                    key = (t, theme)
                    prev = hits.get(key)
                    w = 0.55
                    if prev is None or w > prev.weight:
                        hits[key] = MatchHit(t, theme, kw, w)
                break  # one keyword enough per theme

    # Named companies + govt investment language → strong direct hits
    if govt_hit and company_tickers:
        for t in company_tickers:
            key = (t, "govt_equity")
            prev = hits.get(key)
            w = 1.15
            if prev is None or w > prev.weight:
                hits[key] = MatchHit(t, "govt_equity", govt_hit, w)

    return list(hits.values())
