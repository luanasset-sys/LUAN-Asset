#!/usr/bin/env python3
"""Equity Research HQ — one command, four lenses.

    python hq.py TICKER [--no-airtable] [--no-sheets] [--dry-run] [--no-llm]

Analyzes a stock through four independent lenses (fundamental, technical, Markov
regime, macro/government), writes the results to Airtable + a Google Sheet, and
prints a one-page scorecard that flags where the lenses agree vs disagree.

Decision-support only — not financial advice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import config
import scorecard as scorecard_mod
from data import llm
from data.prices import fetch_market_data
from lenses import fundamental, macro, markov, technical
from models import NA, LensResult


def _safe_lens(name: str, fn, *args) -> LensResult:
    """Run a lens; never let one failure abort the whole run."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        return LensResult(name, "Insufficient Data", NA,
                          f"{name} lens crashed: {exc}", error=str(exc))


def _polish_fundamental(result: LensResult, ticker: str) -> None:
    """Optional Claude rewrite of the fundamental note (kept faithful to metrics)."""
    if result.error or not llm.available():
        return
    prompt = (
        f"Rewrite this equity research note for {ticker} as 2-3 tight, plain-English "
        f"sentences for a sophisticated investor. Do not invent numbers; only use what's "
        f"here. Keep the verdict. Note:\n\n{result.summary}"
    )
    polished = llm.synthesize(prompt)
    if polished:
        result.raw["deterministic_summary"] = result.summary
        result.summary = polished


def run_hq(ticker: str, *, use_airtable: bool, use_sheets: bool, use_llm: bool) -> int:
    ticker = ticker.upper().strip()
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    run_id = f"{ticker} {now.strftime('%Y-%m-%d %H:%M:%S')}"

    if not use_llm:
        config.ANTHROPIC_API_KEY = None  # disable the macro web search + polish

    print(f"\n▶ Researching {ticker} …")

    print("  · fetching market data (yfinance)…")
    md = fetch_market_data(ticker)
    if md.price is None and not md.has_history:
        print(f"✖ Could not fetch any market data for {ticker}. Check the symbol.")
        return 1

    print("  · lens 1/4 fundamental (SEC EDGAR)…")
    f = _safe_lens("fundamental", fundamental.run, md)
    _polish_fundamental(f, ticker)
    print("  · lens 2/4 technical (prices)…")
    t = _safe_lens("technical", technical.run, md)
    print("  · lens 3/4 Markov regime…")
    mk = _safe_lens("markov", markov.run, md)
    print("  · lens 4/4 macro/government (web search)…")
    mc = _safe_lens("macro", macro.run, md)

    run = scorecard_mod.assemble(ticker, run_id, timestamp, md.price, f, t, mk, mc)

    # ── Persist raw snapshot locally (the backtest record) ───────────────────
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    snap_path = os.path.join(config.RUNS_DIR, f"{run_id.replace(':', '').replace(' ', '_')}.json")
    with open(snap_path, "w") as fh:
        json.dump(run.as_dict(), fh, indent=2, default=str)
    print(f"  · saved snapshot → {snap_path}")

    # ── Sinks ────────────────────────────────────────────────────────────────
    if use_airtable:
        try:
            from sinks import airtable_sink
            ids = airtable_sink.write_run(run)
            ok = sum(1 for v in ids.values() if v)
            print(f"  · Airtable: wrote {ok}/{len(ids)} tables (base {config.AIRTABLE_BASE_ID}).")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Airtable skipped: {exc}")

    if use_sheets:
        try:
            from sinks import sheets_sink
            if sheets_sink.configured():
                sheets_sink.write_scorecard(run)
                print("  · Google Sheet: Scorecard row appended.")
            else:
                print("  · Google Sheet skipped (set GOOGLE_SHEET_ID + service-account JSON).")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Google Sheet skipped: {exc}")

    print("\n" + scorecard_mod.render(run, config.DISCLAIMER) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Equity Research HQ — multi-lens stock analysis.")
    p.add_argument("ticker", help="Stock ticker, e.g. NVDA")
    p.add_argument("--no-airtable", action="store_true", help="skip Airtable writes")
    p.add_argument("--no-sheets", action="store_true", help="skip Google Sheets mirror")
    p.add_argument("--no-llm", action="store_true", help="disable Anthropic web search + polish")
    p.add_argument("--dry-run", action="store_true", help="analyze + print only; no sinks")
    args = p.parse_args(argv)

    use_airtable = not (args.no_airtable or args.dry_run)
    use_sheets = not (args.no_sheets or args.dry_run)
    return run_hq(args.ticker, use_airtable=use_airtable, use_sheets=use_sheets,
                  use_llm=not args.no_llm)


if __name__ == "__main__":
    sys.exit(main())
