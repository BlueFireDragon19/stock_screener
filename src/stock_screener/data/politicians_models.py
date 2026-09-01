from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


@dataclass
class PoliticianTrade:
    source: str  # capitoltrades | house | senate
    ticker: str
    politician: str
    chamber: str
    party: str
    tx_type: str  # buy | sell | exchange | other
    trade_date: date | None
    filed_date: date | None
    filed_after_days: int | None
    size_label: str
    size_mid: float
    owner: str
    detail_url: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trade_date"] = self.trade_date.isoformat() if self.trade_date else None
        d["filed_date"] = self.filed_date.isoformat() if self.filed_date else None
        return d


@dataclass
class PoliticianTickerScore:
    ticker: str
    score: float
    buy_weight: float
    sell_weight: float
    trade_count: int
    buy_count: int
    sell_count: int
    avg_filed_after: float | None
    politicians: str
    sources: str
    why: str
    pass_filters: bool
    filter_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# Disclosed STOCK Act amount-band midpoints (USD)
SIZE_MIDPOINTS = {
    "<1k": 500.0,
    "1k-15k": 8_000.0,
    "1k–15k": 8_000.0,
    "15k-50k": 32_500.0,
    "15k–50k": 32_500.0,
    "50k-100k": 75_000.0,
    "50k–100k": 75_000.0,
    "100k-250k": 175_000.0,
    "100k–250k": 175_000.0,
    "250k-500k": 375_000.0,
    "250k–500k": 375_000.0,
    "500k-1m": 750_000.0,
    "500k–1m": 750_000.0,
    "1m-5m": 3_000_000.0,
    "1m–5m": 3_000_000.0,
    "5m-25m": 15_000_000.0,
    "5m–25m": 15_000_000.0,
    "25m-50m": 37_500_000.0,
    "25m–50m": 37_500_000.0,
    "$1,001 - $15,000": 8_000.0,
    "$1,001-$15,000": 8_000.0,
    "$15,001 - $50,000": 32_500.0,
    "$15,001-$50,000": 32_500.0,
    "$50,001 - $100,000": 75_000.0,
    "$50,001-$100,000": 75_000.0,
    "$100,001 - $250,000": 175_000.0,
    "$100,001-$250,000": 175_000.0,
    "$250,001 - $500,000": 375_000.0,
    "$250,001-$500,000": 375_000.0,
    "$500,001 - $1,000,000": 750_000.0,
    "$500,001-$1,000,000": 750_000.0,
    "$1,000,001 - $5,000,000": 3_000_000.0,
    "$1,000,001-$5,000,000": 3_000_000.0,
}


def size_midpoint(label: str) -> float:
    if not label:
        return 8_000.0
    key = label.strip().lower().replace(" ", "")
    # normalize en-dash
    key = key.replace("–", "-").replace("—", "-")
    for k, v in SIZE_MIDPOINTS.items():
        if k.replace(" ", "").replace("–", "-") == key:
            return v
    # fuzzy contains
    compact = label.lower().replace(" ", "").replace("–", "-").replace(",", "")
    if "100k" in compact and "250k" in compact:
        return 175_000.0
    if "1k" in compact and "15k" in compact:
        return 8_000.0
    if "15k" in compact and "50k" in compact:
        return 32_500.0
    if "50k" in compact and "100k" in compact:
        return 75_000.0
    return 8_000.0


def normalize_tx(raw: str) -> str:
    t = (raw or "").strip().lower()
    if t.startswith("buy") or t in {"p", "purchase"}:
        return "buy"
    if t.startswith("sell") or t in {"s", "sale"}:
        return "sell"
    if "exchange" in t:
        return "exchange"
    return "other"


def normalize_ticker(raw: str) -> str:
    t = (raw or "").strip().upper()
    if not t or t in {"N/A", "NA", "NONE"}:
        return ""
    # AMAT:US → AMAT
    if ":" in t:
        t = t.split(":", 1)[0]
    t = t.replace(".", "-")
    return t
