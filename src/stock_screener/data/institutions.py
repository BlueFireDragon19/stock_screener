"""Soft institutional overlay from curated 13F filers (SEC EDGAR, free).

Not a separate mode — blended into support/catalyst social with why tags like
``13F: Berkshire, Pershing (+2)``. Holdings are quarterly and lagged (~45d).
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# SEC fair-access: User-Agent must include a contact email or requests get 403.
USER_AGENT = os.environ.get(
    "SCREENER_SEC_USER_AGENT",
    "stock-screener/0.1 research@example.com",
)

# Curated quality / well-followed 13F managers (name → CIK, zero-padded).
DEFAULT_MANAGERS: dict[str, str] = {
    "Berkshire": "0001067983",
    "Bridgewater": "0001350694",
    "Pershing": "0001336528",
    "Appaloosa": "0001656456",
    "Baupost": "0001061768",
    "TigerGlobal": "0001167483",
    "Coatue": "0001135730",
    "Citadel": "0001423053",
    "Renaissance": "0001037389",
    "Elliott": "0001791786",
}

# Help match 13F issuer strings to tickers.
ISSUER_ALIASES: dict[str, str] = {
    "berkshire hathaway": "BRK-B",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta platforms": "META",
    "facebook": "META",
    "amazon.com": "AMZN",
    "amazon com": "AMZN",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "exxon": "XOM",
    "unitedhealth": "UNH",
    "eli lilly": "LLY",
    "broadcom": "AVGO",
    "salesforce": "CRM",
    "advanced micro devices": "AMD",
}


@dataclass
class InstitutionAssessment:
    score: float  # 0–100
    n_managers: int
    managers: tuple[str, ...]
    total_value_k: float  # 13F values are typically $ thousands
    reason: str

    @property
    def used(self) -> bool:
        return self.n_managers > 0


def _local(tag: str) -> str:
    return tag.split("}")[-1].lower() if tag else ""


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for noise in (
        " inc",
        " corp",
        " corporation",
        " co",
        " ltd",
        " plc",
        " class a",
        " class b",
        " class c",
        " ordinary shares",
        " common stock",
        " com",
    ):
        if t.endswith(noise):
            t = t[: -len(noise)].strip()
    return t


def _sec_get(url: str, *, timeout: float = 30.0) -> requests.Response | None:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("SEC %s → HTTP %s", url, resp.status_code)
            return None
        return resp
    except Exception as exc:  # noqa: BLE001
        logger.debug("SEC request failed %s: %s", url, exc)
        return None


def load_issuer_ticker_map(known_tickers: set[str]) -> dict[str, str]:
    """Map normalized issuer / company title → ticker for known universe."""
    out: dict[str, str] = {}
    for alias, ticker in ISSUER_ALIASES.items():
        if ticker in known_tickers or ticker.replace("-", ".") in known_tickers:
            out[alias] = ticker if ticker in known_tickers else ticker.replace("-", ".")
    resp = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not resp:
        return out
    try:
        rows = resp.json()
    except Exception:  # noqa: BLE001
        return out
    known_u = {t.upper() for t in known_tickers}
    for row in rows.values() if isinstance(rows, dict) else []:
        ticker = str(row.get("ticker") or "").upper().replace(".", "-")
        alt = ticker.replace("-", ".")
        if ticker not in known_u and alt not in known_u:
            continue
        use = ticker if ticker in known_u else alt
        title = _norm(str(row.get("title") or ""))
        if title:
            out.setdefault(title, use)
        # First token / shortened forms
        parts = title.split()
        if parts:
            out.setdefault(parts[0], use)
    return out


def _match_ticker(issuer: str, name_map: dict[str, str]) -> str | None:
    n = _norm(issuer)
    if not n:
        return None
    if n in name_map:
        return name_map[n]
    for alias, ticker in name_map.items():
        if len(alias) >= 4 and (alias in n or n in alias):
            return ticker
    return None


def _latest_13f(cik: str) -> tuple[str, str] | None:
    """Return (accession_no_dashes, filing_date) for latest 13F-HR."""
    resp = _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not resp:
        return None
    try:
        recent = resp.json()["filings"]["recent"]
    except Exception:  # noqa: BLE001
        return None
    forms = recent.get("form") or []
    for i, form in enumerate(forms):
        if str(form).upper() in {"13F-HR", "13F-HR/A"}:
            acc = str(recent["accessionNumber"][i]).replace("-", "")
            date = str(recent["filingDate"][i])
            return acc, date
    return None


def _parse_infotable(xml_bytes: bytes) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    rows: list[dict] = []
    for node in root.iter():
        if _local(node.tag) != "infotable":
            continue
        rec: dict = {}
        for ch in node.iter():
            ln = _local(ch.tag)
            v = (ch.text or "").strip()
            if ln == "nameofissuer":
                rec["issuer"] = v
            elif ln == "cusip":
                rec["cusip"] = v.upper()
            elif ln == "value":
                try:
                    rec["value"] = float(v or 0)
                except ValueError:
                    rec["value"] = 0.0
            elif ln == "sshprnamt":
                try:
                    rec["shares"] = float(v or 0)
                except ValueError:
                    rec["shares"] = 0.0
            elif ln == "putcall":
                rec["putcall"] = v
        if rec.get("putcall"):
            continue  # skip options overlays
        if rec.get("issuer"):
            rows.append(rec)
    return rows


def fetch_manager_holdings(
    manager: str,
    cik: str,
    *,
    pause: float = 0.15,
) -> tuple[list[dict], str]:
    """Holdings list + filing date for one manager's latest 13F."""
    time.sleep(pause)
    latest = _latest_13f(cik)
    if not latest:
        return [], ""
    acc, filed = latest
    time.sleep(pause)
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
    idx = _sec_get(f"{base}/index.json")
    if not idx:
        return [], filed
    try:
        items = idx.json()["directory"]["item"]
    except Exception:  # noqa: BLE001
        return [], filed
    xmls = [it["name"] for it in items if str(it.get("name", "")).lower().endswith(".xml")]
    for name in xmls:
        low = name.lower()
        if "primary" in low or "xslform" in low:
            continue
        time.sleep(pause)
        raw = _sec_get(f"{base}/{name}")
        if not raw:
            continue
        holdings = _parse_infotable(raw.content)
        if holdings:
            logger.info("13F %s (%s): %d lines filed %s", manager, cik, len(holdings), filed)
            return holdings, filed
    return [], filed


def build_institution_assessments(
    known_tickers: set[str],
    *,
    managers: dict[str, str] | None = None,
    pause: float = 0.15,
) -> dict[str, InstitutionAssessment]:
    """
    Scan curated 13F filers once; return per-ticker assessments for ``known_tickers``.
    """
    managers = managers or DEFAULT_MANAGERS
    name_map = load_issuer_ticker_map(known_tickers)
    # ticker → list of (manager, value)
    hits: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for manager, cik in managers.items():
        holdings, _filed = fetch_manager_holdings(manager, cik, pause=pause)
        if not holdings:
            continue
        # Rank by value for "large position" bonus
        ranked = sorted(holdings, key=lambda h: float(h.get("value") or 0), reverse=True)
        top_n = {id(h) for h in ranked[:15]}
        for h in holdings:
            ticker = _match_ticker(str(h.get("issuer") or ""), name_map)
            if not ticker or ticker not in known_tickers:
                # try hyphen/dot variants
                if ticker and ticker.replace("-", ".") in known_tickers:
                    ticker = ticker.replace("-", ".")
                elif ticker and ticker.replace(".", "-") in known_tickers:
                    ticker = ticker.replace(".", "-")
                else:
                    continue
            value = float(h.get("value") or 0.0)
            if id(h) in top_n:
                value *= 1.15  # mild large-position nudge
            hits[ticker].append((manager, value))

    out: dict[str, InstitutionAssessment] = {}
    for ticker in known_tickers:
        rows = hits.get(ticker) or []
        if not rows:
            out[ticker] = InstitutionAssessment(
                score=50.0,
                n_managers=0,
                managers=(),
                total_value_k=0.0,
                reason="13F: n/a",
            )
            continue
        # Dedupe managers (keep max value)
        by_mgr: dict[str, float] = {}
        for mgr, val in rows:
            by_mgr[mgr] = max(by_mgr.get(mgr, 0.0), val)
        mgrs = tuple(sorted(by_mgr.keys(), key=lambda m: -by_mgr[m]))
        n = len(mgrs)
        total = sum(by_mgr.values())
        score = 50.0 + min(36.0, 12.0 * n)
        if total >= 1_000_000:  # very large aggregate ($ thousands → $1B+)
            score += 8.0
        elif total >= 100_000:
            score += 4.0
        score = max(0.0, min(100.0, score))
        label = ", ".join(mgrs[:4]) + ("…" if n > 4 else "")
        out[ticker] = InstitutionAssessment(
            score=round(score, 1),
            n_managers=n,
            managers=mgrs,
            total_value_k=total,
            reason=f"13F: {label} (+{n})",
        )

    covered = sum(1 for a in out.values() if a.used)
    logger.info("Institutions 13F coverage: %d / %d tickers", covered, len(known_tickers))
    return out
