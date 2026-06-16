"""Mirror the Scorecard to a Google Sheet (phone-friendly).

Uses a service account. See README.md "Google Sheets setup" for the click-by-
click on creating the credentials and sharing the sheet. The sheet/tab and a
header row are created on first run if missing.
"""
from __future__ import annotations

import os

import config
from models import RunResult

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None  # type: ignore

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_TAB = "Scorecard"
_HEADER = [
    "Run Timestamp", "Ticker", "Price",
    "Fundamental", "Technical", "Markov", "Macro",
    "Disagreement", "FV Low", "FV Base", "FV High", "Cross-lens read",
]


def configured() -> bool:
    return bool(config.GOOGLE_SHEET_ID and os.path.exists(config.GOOGLE_SA_JSON))


def write_scorecard(run: RunResult) -> bool:
    if gspread is None:
        raise RuntimeError("gspread/google-auth not installed.")
    if not config.GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID not set.")
    if not os.path.exists(config.GOOGLE_SA_JSON):
        raise RuntimeError(f"Service-account JSON not found at {config.GOOGLE_SA_JSON}.")

    creds = Credentials.from_service_account_file(config.GOOGLE_SA_JSON, scopes=_SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(config.GOOGLE_SHEET_ID)

    try:
        ws = sh.worksheet(_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=_TAB, rows=1000, cols=len(_HEADER))

    if ws.row_count == 0 or ws.acell("A1").value != _HEADER[0]:
        ws.update("A1", [_HEADER])

    cross_lens = run.agreement_map.splitlines()[0] if run.agreement_map else ""
    row = [
        run.timestamp, run.ticker,
        _num(run.price),
        run.fundamental.verdict, run.technical.verdict,
        run.markov.metrics.get("Current State", run.markov.verdict),
        run.macro.verdict,
        "YES" if run.disagreement else "no",
        _num(run.fair_value.get("low")), _num(run.fair_value.get("base")),
        _num(run.fair_value.get("high")), cross_lens,
    ]
    ws.append_row(row, value_input_option="USER_ENTERED")
    return True


def _num(x):
    return "" if x is None else round(float(x), 2)
