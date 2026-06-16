#!/usr/bin/env python3
"""Equity Research HQ — one command, four lenses.

    python hq.py TICKER          [--no-airtable] [--no-sheets] [--dry-run] [--no-llm]
    python hq.py --watchlist LIST [same flags]      # LIST = file path OR NVDA,VST,CEG

Analyzes a stock (or a whole watchlist) through four independent lenses
(fundamental, technical, Markov regime, macro/government), writes the results to
Airtable + a Google Sheet, and prints a one-page scorecard per ticker plus — for
a watchlist — a side-by-side comparison table.

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
from models import NA, LensResult, RunResult


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


def analyze(ticker: str) -> RunResult | None:
    """Run all four lenses for one ticker and assemble the scorecard.

    Returns None only when no market data could be fetched at all.
    Always writes the local JSON snapshot (the backtest record).
    """
    ticker = ticker.upper().strip()
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    run_id = f"{ticker} {now.strftime('%Y-%m-%d %H:%M:%S')}"

    print(f"\n▶ Researching {ticker} …")
    print("  · fetching market data (yfinance)…")
    md = fetch_market_data(ticker)
    if md.price is None and not md.has_history:
        print(f"  ✖ No market data for {ticker}. Skipping.")
        return None

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

    os.makedirs(config.RUNS_DIR, exist_ok=True)
    snap_path = os.path.join(config.RUNS_DIR, f"{run_id.replace(':', '').replace(' ', '_')}.json")
    with open(snap_path, "w") as fh:
        json.dump(run.as_dict(), fh, indent=2, default=str)
    print(f"  · saved snapshot → {snap_path}")
    return run


def persist(run: RunResult, *, use_airtable: bool, use_sheets: bool) -> None:
    """Write a completed run to the configured sinks; failures are non-fatal."""
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


def parse_watchlist(value: str) -> list[str]:
    """A file path (one ticker per line, '#' comments ok) OR an inline list
    separated by commas/whitespace. De-duplicated, order preserved."""
    if os.path.isfile(value):
        with open(value) as fh:
            raw = []
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    raw.extend(line.replace(",", " ").split())
    else:
        raw = value.replace(",", " ").split()
    seen, out = set(), []
    for tk in raw:
        u = tk.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def run_single(ticker: str, *, use_airtable: bool, use_sheets: bool) -> int:
    run = analyze(ticker)
    if run is None:
        return 1
    persist(run, use_airtable=use_airtable, use_sheets=use_sheets)
    print("\n" + scorecard_mod.render(run, config.DISCLAIMER) + "\n")
    return 0


def run_watchlist(tickers: list[str], *, use_airtable: bool, use_sheets: bool,
                  show_cards: bool) -> int:
    if not tickers:
        print("✖ Empty watchlist.")
        return 1
    print(f"▶ Watchlist: {', '.join(tickers)}")
    runs: list[RunResult] = []
    for tk in tickers:
        run = analyze(tk)
        if run is None:
            continue
        persist(run, use_airtable=use_airtable, use_sheets=use_sheets)
        if show_cards:
            print("\n" + scorecard_mod.render(run, config.DISCLAIMER) + "\n")
        runs.append(run)

    print("\n" + scorecard_mod.render_comparison(runs, config.DISCLAIMER) + "\n")
    return 0 if runs else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Equity Research HQ — multi-lens stock analysis.")
    p.add_argument("ticker", nargs="?", help="Stock ticker, e.g. NVDA")
    p.add_argument("--watchlist", metavar="LIST",
                   help="file path (one ticker per line) OR inline list e.g. NVDA,VST,CEG")
    p.add_argument("--full", action="store_true",
                   help="with --watchlist, also print each ticker's full scorecard")
    p.add_argument("--no-airtable", action="store_true", help="skip Airtable writes")
    p.add_argument("--no-sheets", action="store_true", help="skip Google Sheets mirror")
    p.add_argument("--no-llm", action="store_true", help="disable Anthropic web search + polish")
    p.add_argument("--dry-run", action="store_true", help="analyze + print only; no sinks")
    args = p.parse_args(argv)

    if not args.ticker and not args.watchlist:
        p.error("provide a TICKER or --watchlist LIST")
    if args.ticker and args.watchlist:
        p.error("use either a single TICKER or --watchlist, not both")

    if args.no_llm:
        config.ANTHROPIC_API_KEY = None  # disable macro web search + polish

    use_airtable = not (args.no_airtable or args.dry_run)
    use_sheets = not (args.no_sheets or args.dry_run)

    if args.watchlist:
        return run_watchlist(parse_watchlist(args.watchlist),
                             use_airtable=use_airtable, use_sheets=use_sheets,
                             show_cards=args.full)
    return run_single(args.ticker, use_airtable=use_airtable, use_sheets=use_sheets)


if __name__ == "__main__":
    sys.exit(main())
