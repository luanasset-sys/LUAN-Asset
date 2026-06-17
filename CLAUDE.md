# CLAUDE.md — Equity Research HQ

Architecture notes and standing rules for this project. Read this before changing code.

## What this is

A personal, re-runnable equity research harness:

```
python hq.py TICKER
```

analyzes a stock through **four independent lenses**, writes results to Airtable
and a Google Sheet, and prints a one-page scorecard. The point is **multiple
independent views, not one number** — where the lenses disagree is the signal.

## Standing rules (do not break these)

1. **Decision-support, not financial advice.** The disclaimer line
   (`config.DISCLAIMER`) appears in the printed scorecard and the Airtable
   Scorecard row. Keep it.
2. **Don't collapse to a single buy/sell score.** The scorecard shows four
   verdicts side by side and an explicit AGREE/DISAGREE map. Preserve that.
3. **Keep raw data + timestamps.** Every run writes a full JSON snapshot to
   `runs/` and a `Raw JSON` column to each Airtable detail table, so everything
   is backtestable later. Never drop the raw layer.
4. **Start free.** Data sources are SEC EDGAR + yfinance + FRED only. The
   Anthropic API powers the macro web search and optional narrative polish. No
   paid market-data vendors unless explicitly requested.
5. **Lenses are independent and isolated.** One lens failing must never abort the
   run (`hq._safe_lens`). A lens with missing data returns
   `verdict="Insufficient Data", stance="na"` — it does not raise.
6. **The Markov lens is REGIME FRAMING, not price prediction.** Its output text
   must say so and must not overstate confidence.

## Layout

```
hq.py            CLI orchestrator: single ticker OR --watchlist; per-ticker
                 `analyze()` → `persist()` → render; watchlist adds a comparison table
app.py           Streamlit dashboard: interactive DCF (sliders → live fair value +
                 sensitivity grid), financials tables, charts, side-by-side compare.
                 Driven by `fundamental.dcf_per_share` + `fundamental.model_detail`.
config.py        env, benchmarks, sector→ETF map, thresholds, DCF assumptions
models.py        LensResult / RunResult dataclasses + stance constants

data/            thin clients (no business logic)
  edgar.py       SEC EDGAR XBRL company-facts → annual series (proper User-Agent)
  prices.py      yfinance: ticker + SPY + sector ETF, one shared fetch
  fred.py        FRED rates/CPI (free key)
  llm.py         Anthropic wrapper: web_search research + narrative synthesis

lenses/          the four lenses, each `run(md) -> LensResult`
  fundamental.py DCF (bull/base/bear) + multiples vs OWN 5y history + quality + growth
  technical.py   MAs/cross, RSI, 3/6/12mo returns, vol, drawdown, RS vs SPY & sector
  markov.py      weekly Up/Flat/Down × hi/lo-vol states → transition matrix → steady state
  macro.py       FRED posture + Claude web search → tailwind/headwind, cited & dated

sinks/
  airtable_sink.py  Scorecard + 4 detail tables (field names match the base schema)
  sheets_sink.py    appends a Scorecard row to a Google Sheet (service account)

scorecard.py     assembles the 4 verdicts, computes the agree/disagree map, renders
runs/            local JSON snapshots (gitignored contents) — the backtest record
```

## Data model (Airtable base "Equity Research HQ", `appL5ByzBTesKLEWo`)

- **Scorecard** — one row per ticker per run: the four verdicts + summaries,
  agreement map, disagreement flag, price, fair-value range, timestamp.
- **Fundamentals / Technicals / Markov / Macro Signals** — one row per run each,
  holding the computed metrics + a `Raw JSON` snapshot.
- Percent fields store **fractions** (0.25 == 25%); the lenses emit fractions.
- The Google Sheet mirrors the **Scorecard** tab only.

## Conventions / gotchas

- **Field names are the contract.** `airtable_sink` maps `LensResult.metrics`
  keys straight to Airtable field names — they must match the base schema. If you
  rename a metric, rename the field (or the write silently drops it after
  `typecast`).
- **Percent vs fraction.** Lenses return fractions everywhere. FRED returns
  percent; `macro._as_frac` converts before storing.
- **EDGAR tagging varies by filer.** `edgar.CONCEPT_TAGS` lists fallback tags per
  concept. Add to those lists rather than hard-coding one tag.
- **"Revenue TTM"** currently stores the latest fiscal-year value (annual model);
  upgrade to a true trailing-twelve-month sum if/when quarterly parsing is added.
- **Markov HMM upgrade hook:** `markov.classify_states` /
  `estimate_transition_matrix` are separated so a `hmmlearn` latent-state model
  can replace the threshold classifier — see `markov.fit_hmm_states`.

## Credentials (see `.env.example`)

`SEC_USER_AGENT` (contact string, no key), `FRED_API_KEY`, `ANTHROPIC_API_KEY`,
`AIRTABLE_API_KEY` (+ `AIRTABLE_BASE_ID`), `GOOGLE_SA_JSON` + `GOOGLE_SHEET_ID`.

## Model usage

The macro lens and narrative polish use `claude-opus-4-8` (`config.ANTHROPIC_MODEL`)
with the server-side `web_search_20260209` tool. The web-search turn handles the
`pause_turn` server-tool loop in `data/llm.py`.
