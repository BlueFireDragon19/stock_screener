from __future__ import annotations

import logging
import re
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from stock_screener.data.politicians_models import (
    PoliticianTrade,
    normalize_ticker,
    normalize_tx,
    size_midpoint,
)

logger = logging.getLogger(__name__)

HOME = "https://efdsearch.senate.gov/search/home/"
UA = {
    "User-Agent": "Mozilla/5.0 stock-screener/0.1 research (STOCK Act public data)",
    "Accept": "text/html,application/xhtml+xml",
}


def _parse_date(text: str):
    text = (text or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def fetch_senate_trades(max_rows: int = 80) -> list[PoliticianTrade]:
    """
    Best-effort Senate Electronic Financial Disclosure search.

    Senate EFD often requires interactive CSRF/agreement flows; when that
    blocks automation we return [] and rely on Capitol Trades + House.
    """
    session = requests.Session()
    session.headers.update(UA)
    out: list[PoliticianTrade] = []
    try:
        home = session.get(HOME, timeout=45)
        if home.status_code != 200:
            logger.warning("Senate EFD home → HTTP %s", home.status_code)
            return []
        soup = BeautifulSoup(home.text, "html.parser")
        # Agree checkbox / token patterns vary; try common PTR search report URL
        # Fallback: parse any report links already visible after landing.
        token = None
        for inp in soup.find_all("input"):
            name = (inp.get("name") or "").lower()
            if "csrf" in name or "token" in name:
                token = inp.get("value")
                break

        # Attempt PTR report listing endpoint used by some scrapers
        candidates = [
            "https://efdsearch.senate.gov/search/report/ptr/",
            "https://efdsearch.senate.gov/search/",
        ]
        html = home.text
        for url in candidates:
            try:
                data = {}
                if token:
                    data["csrfmiddlewaretoken"] = token
                resp = session.post(url, data=data, timeout=45, headers={"Referer": HOME})
                if resp.status_code == 200 and len(resp.text) > 1000:
                    html = resp.text
                    break
            except Exception:  # noqa: BLE001
                continue

        soup = BeautifulSoup(html, "html.parser")
        # Look for table rows that resemble filings
        for tr in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 3:
                continue
            joined = " | ".join(cells)
            if "Periodic Transaction Report" not in joined and "PTR" not in joined:
                # Also accept rows with ticker-looking tokens + Purchase/Sale
                if not re.search(r"\b(Purchase|Sale|Buy|Sell)\b", joined, re.I):
                    continue
            politician = cells[0]
            # Heuristic ticker extraction
            tickers = re.findall(r"\b([A-Z]{1,5})\b", joined)
            tx = "other"
            if re.search(r"\b(Purchase|Buy)\b", joined, re.I):
                tx = "buy"
            elif re.search(r"\b(Sale|Sell)\b", joined, re.I):
                tx = "sell"
            dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", joined)
            trade_d = _parse_date(dates[0]) if dates else None
            filed_d = _parse_date(dates[1]) if len(dates) > 1 else None
            filed_after = None
            if trade_d and filed_d:
                filed_after = max(0, (filed_d - trade_d).days)
            amt = ""
            m_amt = re.search(r"\$[\d,]+\s*-\s*\$[\d,]+", joined)
            if m_amt:
                amt = m_amt.group(0)
            for t in tickers:
                nt = normalize_ticker(t)
                if not nt or nt in {"PTR", "PDF", "HTTP", "HTTPS", "REPORT"}:
                    continue
                out.append(
                    PoliticianTrade(
                        source="senate",
                        ticker=nt,
                        politician=politician,
                        chamber="Senate",
                        party="",
                        tx_type=normalize_tx(tx),
                        trade_date=trade_d,
                        filed_date=filed_d,
                        filed_after_days=filed_after,
                        size_label=amt or "1K–15K",
                        size_mid=size_midpoint(amt or "1K–15K"),
                        owner="",
                        detail_url=urljoin(HOME, ""),
                    )
                )
                if len(out) >= max_rows:
                    break
            if len(out) >= max_rows:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Senate EFD fetch failed: %s", exc)
        return []

    logger.info("Senate official (best-effort): %d trades", len(out))
    return out
