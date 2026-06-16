"""SEC EDGAR client — free, no API key, but a proper User-Agent is mandatory.

Pulls XBRL company-facts and exposes tidy annual (fiscal-year) series for the
fundamental model. Everything degrades gracefully: missing concepts return {}
rather than raising, so the lens can report "insufficient data".
"""
from __future__ import annotations

import functools
import time
from typing import Any

import requests

import config

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Candidate us-gaap tags per concept, tried in order (filers tag inconsistently).
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "total_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "current_debt": ["LongTermDebtCurrent", "DebtCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}

# Balance-sheet concepts are point-in-time (instant); income/cash-flow are flows.
_INSTANT_CONCEPTS = {"total_debt", "current_debt", "cash", "equity"}


def _headers() -> dict[str, str]:
    return {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _get(url: str, retries: int = 3) -> dict[str, Any] | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            # 403/429 -> back off (EDGAR rate-limits aggressively)
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


@functools.lru_cache(maxsize=1)
def _ticker_cik_map() -> dict[str, int]:
    data = _get(_TICKERS_URL) or {}
    out: dict[str, int] = {}
    for row in data.values():
        out[str(row["ticker"]).upper()] = int(row["cik_str"])
    return out


def get_cik(ticker: str) -> int | None:
    return _ticker_cik_map().get(ticker.upper())


@functools.lru_cache(maxsize=32)
def get_company_facts(cik: int) -> dict[str, Any] | None:
    return _get(_FACTS_URL.format(cik=cik))


def _annual_series(facts: dict[str, Any], concept: str) -> dict[int, float]:
    """Return {fiscal_year: value} from 10-K / FY annual filings for a concept.

    Picks the entry whose period matches the fiscal year and prefers
    framed annual values, deduping to the latest-filed value per year.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    instant = concept in _INSTANT_CONCEPTS
    for tag in CONCEPT_TAGS.get(concept, []):
        node = gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        # USD for $ concepts, USD/shares for EPS.
        unit_key = "USD/shares" if concept == "eps_diluted" else "USD"
        rows = units.get(unit_key)
        if not rows:
            continue
        by_year: dict[int, tuple[str, float]] = {}  # fy -> (end_date, val)
        for r in rows:
            if r.get("form") not in ("10-K", "10-K/A", "20-F"):
                continue
            fy = r.get("fy")
            fp = r.get("fp")
            if fy is None or fp != "FY":
                continue
            # For flow concepts, only accept full-year periods (~360+ days).
            if not instant:
                start, end = r.get("start"), r.get("end")
                if start and end:
                    days = (_to_days(end) - _to_days(start))
                    if days < 350:
                        continue
            end = r.get("end", "")
            val = r.get("val")
            if val is None:
                continue
            prev = by_year.get(int(fy))
            if prev is None or end >= prev[0]:
                by_year[int(fy)] = (end, float(val))
        if by_year:
            return {fy: v for fy, (_, v) in by_year.items()}
    return {}


def _to_days(date_str: str) -> int:
    y, m, d = (int(x) for x in date_str.split("-"))
    return y * 365 + m * 30 + d


def fundamentals_annual(ticker: str) -> dict[str, Any]:
    """Top-level entry point. Returns annual series keyed by concept plus meta.

    Shape: {"cik": int, "fiscal_years": [...], "series": {concept: {fy: val}}}
    """
    cik = get_cik(ticker)
    if cik is None:
        return {"error": f"No SEC CIK found for {ticker} (foreign/ETF/private?)."}
    facts = get_company_facts(cik)
    if not facts:
        return {"error": f"EDGAR company-facts unavailable for CIK {cik}."}

    series = {concept: _annual_series(facts, concept) for concept in CONCEPT_TAGS}
    years = sorted({fy for s in series.values() for fy in s})
    return {
        "cik": cik,
        "entity": facts.get("entityName"),
        "fiscal_years": years,
        "series": series,
    }
