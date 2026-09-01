from __future__ import annotations

import csv
import io
import logging
import re
import time
import zipfile
from datetime import date, datetime
from io import StringIO

import requests

from stock_screener.data.politicians_models import (
    PoliticianTrade,
    normalize_ticker,
    normalize_tx,
    size_midpoint,
)

logger = logging.getLogger(__name__)

UA = {
    "User-Agent": "stock-screener/0.1 (+local research; STOCK Act public disclosures)"
}
TICKER_RE = re.compile(
    r"\(([A-Z][A-Z0-9.\-]{0,9})\)\s*\[ST\]|"
    r"\b([A-Z]{1,5})\b\s*\[ST\]|"
    r"Common Stock\s*\(([A-Z][A-Z0-9.\-]{0,9})\)|"
    r"\(([A-Z][A-Z0-9.\-]{0,9})\)\s*$",
    re.I,
)
TX_LINE_RE = re.compile(
    r"(?P<asset>.+?)\s+(?P<tx>P|S|E)(?:\s*\([^)]*\))?\s+"
    r"(?P<traded>\d{2}/\d{2}/\d{4})\s+(?P<notified>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<amount>\$[\d,]+\s*-\s*\$[\d,]+)",
    re.S,
)


def _years() -> list[int]:
    y = datetime.utcnow().year
    return [y, y - 1]


def _parse_mdy(s: str):
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def _load_ptr_index(year: int) -> list[dict]:
    url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
    resp = requests.get(url, headers=UA, timeout=60)
    if resp.status_code != 200:
        logger.warning("House FD zip %s → HTTP %s", year, resp.status_code)
        return []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    txt_name = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
    if not txt_name:
        return []
    text = zf.read(txt_name).decode("utf-8", errors="replace")
    rows = list(csv.DictReader(StringIO(text), delimiter="\t"))
    ptrs = [r for r in rows if (r.get("FilingType") or "").strip() == "P"]
    ptrs.sort(
        key=lambda r: _parse_mdy(r.get("FilingDate") or "") or date.min,
        reverse=True,
    )
    logger.info("House %s: %d PTR filings in index", year, len(ptrs))
    return ptrs


def _extract_ticker(asset: str) -> str:
    for m in TICKER_RE.finditer(asset or ""):
        for g in m.groups():
            if g:
                return normalize_ticker(g)
    return ""


def _parse_ptr_pdf(
    content: bytes,
    *,
    politician: str,
    filed_date,
    doc_id: str,
    year: int,
) -> list[PoliticianTrade]:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; skip House PTR parse")
        return []

    trades: list[PoliticianTrade] = []
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    blob = "\n".join(text_parts)
    # Normalize weird nulls from some PDFs
    blob = blob.replace("\x00", "")

    for m in TX_LINE_RE.finditer(blob):
        asset = m.group("asset")
        ticker = _extract_ticker(asset)
        if not ticker:
            continue
        tx = normalize_tx(m.group("tx"))
        traded = _parse_mdy(m.group("traded"))
        notified = _parse_mdy(m.group("notified"))
        amount = m.group("amount")
        filed_after = None
        if traded and filed_date:
            filed_after = max(0, (filed_date - traded).days)
        elif traded and notified:
            filed_after = max(0, (notified - traded).days)
        trades.append(
            PoliticianTrade(
                source="house",
                ticker=ticker,
                politician=politician,
                chamber="House",
                party="",
                tx_type=tx,
                trade_date=traded,
                filed_date=filed_date,
                filed_after_days=filed_after,
                size_label=amount,
                size_mid=size_midpoint(amount),
                owner="",
                detail_url=(
                    f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/"
                    f"{year}/{doc_id}.pdf"
                ),
            )
        )
    return trades


def fetch_house_trades(
    max_filings: int = 40,
    pause: float = 0.35,
) -> list[PoliticianTrade]:
    """Official House Clerk PTR index + recent PDF parses."""
    out: list[PoliticianTrade] = []
    seen_docs: set[str] = set()
    for year in _years():
        try:
            ptrs = _load_ptr_index(year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("House index %s failed: %s", year, exc)
            continue
        for row in ptrs:
            if len(seen_docs) >= max_filings:
                break
            doc = (row.get("DocID") or "").strip()
            if not doc or doc in seen_docs:
                continue
            seen_docs.add(doc)
            politician = f"{(row.get('First') or '').strip()} {(row.get('Last') or '').strip()}".strip()
            filed = _parse_mdy(row.get("FilingDate") or "")
            url = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc}.pdf"
            try:
                resp = requests.get(url, headers=UA, timeout=60)
                if resp.status_code != 200 or b"%PDF" not in resp.content[:16]:
                    logger.debug("PTR %s → HTTP %s", doc, resp.status_code)
                    continue
                trades = _parse_ptr_pdf(
                    resp.content,
                    politician=politician,
                    filed_date=filed,
                    doc_id=doc,
                    year=year,
                )
                out.extend(trades)
            except Exception as exc:  # noqa: BLE001
                logger.debug("PTR parse %s failed: %s", doc, exc)
            time.sleep(pause)
        if len(seen_docs) >= max_filings:
            break
    logger.info("House official: %d parsed stock trades from %d filings", len(out), len(seen_docs))
    return out
