"""Write a full run to Airtable: Scorecard + the four per-lens detail tables.

Field names here MUST match the base schema ("Equity Research HQ"). Percent
fields are stored as fractions (0.25 == 25%), matching how the lenses emit them.
Each table write is isolated so one failure doesn't abort the others.
"""
from __future__ import annotations

import json

import config
from models import RunResult

try:
    from pyairtable import Api
except ImportError:
    Api = None  # type: ignore

SCORECARD = "Scorecard"
FUNDAMENTALS = "Fundamentals"
TECHNICALS = "Technicals"
MARKOV = "Markov"
MACRO = "Macro Signals"


def _clean(fields: dict) -> dict:
    """Drop None / NaN so Airtable doesn't choke on empty typed cells."""
    out = {}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, float) and (v != v):  # NaN
            continue
        out[k] = v
    return out


def _common(run: RunResult) -> dict:
    return {"Run ID": run.run_id, "Ticker": run.ticker, "Run Timestamp": run.timestamp}


def write_run(run: RunResult) -> dict[str, str | None]:
    """Returns {table: record_id or None}. Raises only on missing config."""
    if Api is None:
        raise RuntimeError("pyairtable not installed — `pip install pyairtable`.")
    if not config.AIRTABLE_API_KEY:
        raise RuntimeError("AIRTABLE_API_KEY not set in environment/.env.")

    api = Api(config.AIRTABLE_API_KEY)
    base = config.AIRTABLE_BASE_ID
    results: dict[str, str | None] = {}

    f, t, mk, mc = run.fundamental, run.technical, run.markov, run.macro

    scorecard = {
        **_common(run),
        "Fundamental Verdict": f.verdict,
        "Fundamental Summary": f.summary,
        "Technical Verdict": t.verdict,
        "Technical Summary": t.summary,
        "Markov State": mk.metrics.get("Current State", mk.verdict),
        "Markov Summary": mk.summary,
        "Macro Verdict": mc.verdict,
        "Macro Summary": mc.summary,
        "Agreement Map": run.agreement_map,
        "Disagreement Flag": run.disagreement,
        "Current Price": run.price,
        "Fair Value Low": run.fair_value.get("low"),
        "Fair Value Base": run.fair_value.get("base"),
        "Fair Value High": run.fair_value.get("high"),
        "Disclaimer": config.DISCLAIMER,
    }

    detail_tables = {
        FUNDAMENTALS: {**_common(run), **f.metrics, "Verdict": f.verdict,
                       "Reasoning": f.summary, "Raw JSON": _dump(f.raw)},
        TECHNICALS: {**_common(run), **t.metrics, "Verdict": t.verdict,
                     "Reasoning": t.summary, "Raw JSON": _dump(t.raw)},
        MARKOV: {**_common(run), **mk.metrics, "Raw JSON": _dump(mk.raw)},
        MACRO: {**_common(run), **mc.metrics, "Raw JSON": _dump(mc.raw)},
    }

    for table_name, fields in [(SCORECARD, scorecard), *detail_tables.items()]:
        try:
            rec = api.table(base, table_name).create(_clean(fields), typecast=True)
            results[table_name] = rec.get("id")
        except Exception as exc:  # noqa: BLE001 — report, keep going
            results[table_name] = None
            print(f"  ! Airtable write to '{table_name}' failed: {exc}")
    return results


def _dump(obj) -> str:
    try:
        return json.dumps(obj, default=str)[:95000]
    except (TypeError, ValueError):
        return str(obj)[:95000]
