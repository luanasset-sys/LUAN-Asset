"""FRED client — latest values for a handful of rate/inflation series.

Free API key required (config.FRED_API_KEY). Returns None gracefully when the
key is absent so the macro lens can still run on web search alone.
"""
from __future__ import annotations

import requests

import config

_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "fed_funds": "DFF",      # effective federal funds rate (daily)
    "ust_10y": "DGS10",      # 10-year treasury
    "ust_2y": "DGS2",        # 2-year treasury
    "cpi_yoy": "CPIAUCSL",   # CPI index (we derive YoY)
}


def _latest(series_id: str, observations: int = 400) -> float | None:
    if not config.FRED_API_KEY:
        return None
    params = {
        "series_id": series_id,
        "api_key": config.FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": observations,
    }
    try:
        resp = requests.get(_OBS_URL, params=params, timeout=20)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
    except requests.RequestException:
        return None
    for o in obs:  # newest first; skip missing "."
        if o.get("value") not in (".", "", None):
            try:
                return float(o["value"])
            except ValueError:
                continue
    return None


def _cpi_yoy() -> float | None:
    """Year-over-year CPI inflation from the index level (12 months apart)."""
    if not config.FRED_API_KEY:
        return None
    params = {
        "series_id": SERIES["cpi_yoy"],
        "api_key": config.FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 13,
    }
    try:
        resp = requests.get(_OBS_URL, params=params, timeout=20)
        resp.raise_for_status()
        obs = [o for o in resp.json().get("observations", []) if o.get("value") not in (".", "", None)]
    except requests.RequestException:
        return None
    if len(obs) < 13:
        return None
    try:
        latest, year_ago = float(obs[0]["value"]), float(obs[12]["value"])
        return (latest / year_ago - 1.0) * 100.0
    except (ValueError, ZeroDivisionError):
        return None


def rates_snapshot() -> dict[str, float | None]:
    """Latest macro rates. Values are in percent (e.g. 4.33 == 4.33%)."""
    return {
        "fed_funds": _latest(SERIES["fed_funds"]),
        "ust_10y": _latest(SERIES["ust_10y"]),
        "ust_2y": _latest(SERIES["ust_2y"]),
        "cpi_yoy": _cpi_yoy(),
    }
