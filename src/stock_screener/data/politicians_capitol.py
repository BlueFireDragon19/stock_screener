from __future__ import annotations

import logging
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup

from stock_screener.data.politicians_models import (
    PoliticianTrade,
    normalize_ticker,
    normalize_tx,
    size_midpoint,
)

logger = logging.getLogger(__name__)

BASE = "https://www.capitoltrades.com"


def _session():
    try:
        from curl_cffi import requests as creq

        s = creq.Session()
        s.impersonate = "chrome"
        return s, True
    except Exception:  # noqa: BLE001
        import requests

        return requests.Session(), False


def _parse_date(text: str):
    text = re.sub(r"\s+", " ", (text or "").strip())
    for fmt in ("%d %b %Y", "%b %d %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_filed_after(text: str) -> int | None:
    m = re.search(r"(\d+)", text or "")
    return int(m.group(1)) if m else None


def _parse_politician_cell(text: str) -> tuple[str, str, str]:
    # "Jared Moskowitz Democrat House FL"
    parts = text.split()
    party = chamber = ""
    name_parts: list[str] = []
    for p in parts:
        low = p.lower()
        if low in {"democrat", "republican", "independent"}:
            party = p
        elif low in {"house", "senate"}:
            chamber = p
        elif re.fullmatch(r"[A-Z]{2}", p):
            continue  # state
        else:
            name_parts.append(p)
    return " ".join(name_parts), chamber, party


def _parse_issuer_cell(text: str) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    m = re.search(r"\b([A-Z][A-Z0-9.\-]{0,9}):US\b", text)
    if m:
        ticker = normalize_ticker(m.group(1))
        name = text[: m.start()].strip()
        return ticker, name
    m2 = re.search(r"\(([A-Z][A-Z0-9.\-]{1,9})\)\s*$", text)
    if m2:
        return normalize_ticker(m2.group(1)), text
    # trailing ticker without :US
    toks = text.split()
    if toks and re.fullmatch(r"[A-Z]{1,5}", toks[-1]):
        return toks[-1], " ".join(toks[:-1])
    return "", text


def fetch_capitol_trades(
    pages: int = 5,
    page_size: int = 96,
    pause: float = 0.4,
) -> list[PoliticianTrade]:
    """Scrape recent trades from capitoltrades.com (public site)."""
    session, using_cffi = _session()
    logger.info(
        "Scraping Capitol Trades (%d pages × %d) via %s",
        pages,
        page_size,
        "curl_cffi" if using_cffi else "requests",
    )
    out: list[PoliticianTrade] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for page in range(1, pages + 1):
        try:
            kwargs = {
                "params": {"page": page, "pageSize": page_size},
                "headers": headers,
                "timeout": 45,
            }
            if using_cffi:
                resp = session.get(f"{BASE}/trades", impersonate="chrome", **kwargs)
            else:
                resp = session.get(f"{BASE}/trades", **kwargs)
            if resp.status_code != 200:
                logger.warning("Capitol Trades page %s → HTTP %s", page, resp.status_code)
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                logger.warning("Capitol Trades page %s: no table", page)
                break
            rows = table.find_all("tr")[1:]
            if not rows:
                break
            for tr in rows:
                cells = tr.find_all("td")
                if len(cells) < 9:
                    continue
                pol_txt = cells[0].get_text(" ", strip=True)
                issuer_txt = cells[1].get_text(" ", strip=True)
                published = _parse_date(cells[2].get_text(" ", strip=True))
                traded = _parse_date(cells[3].get_text(" ", strip=True))
                filed_after = _parse_filed_after(cells[4].get_text(" ", strip=True))
                owner = cells[5].get_text(" ", strip=True)
                tx = normalize_tx(cells[6].get_text(" ", strip=True))
                size_label = cells[7].get_text(" ", strip=True)
                detail = ""
                for a in cells[-1].find_all("a", href=True):
                    if "/trades/" in a["href"]:
                        detail = BASE + a["href"]
                        break
                politician, chamber, party = _parse_politician_cell(pol_txt)
                ticker, _ = _parse_issuer_cell(issuer_txt)
                if not ticker:
                    continue
                if filed_after is None and traded and published:
                    filed_after = max(0, (published - traded).days)
                out.append(
                    PoliticianTrade(
                        source="capitoltrades",
                        ticker=ticker,
                        politician=politician,
                        chamber=chamber,
                        party=party,
                        tx_type=tx,
                        trade_date=traded,
                        filed_date=published,
                        filed_after_days=filed_after,
                        size_label=size_label,
                        size_mid=size_midpoint(size_label),
                        owner=owner,
                        detail_url=detail,
                    )
                )
            time.sleep(pause)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Capitol Trades scrape failed on page %s: %s", page, exc)
            break

    logger.info("Capitol Trades: %d trades with tickers", len(out))
    return out
