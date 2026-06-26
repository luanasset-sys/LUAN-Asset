# reports/

Saved equity deep-dive reports, one markdown file per ticker (`GEV.md`, `NVDA.md`, …).

**The dashboard's "📝 Research Report" tab reads these files directly** — so viewing a
report needs **no Anthropic API key and no credit**. If `reports/<TICKER>.md` exists,
the tab renders it (with a Download button); otherwise it tells you how to get one.

## How a report gets here

These are written in a **Claude Code** session (covered by your existing plan — no
metered API spend). In Claude Code, say:

> "write the deep-dive report for NVDA"

Claude does the live web research, grounds it in this repo's computed model
(bottom-up WACC, multi-stage DCF, reverse-DCF, multiples-vs-own-history), writes the
full institutional-depth note, and saves it to `reports/NVDA.md`. Pull, refresh the
dashboard, and it appears under the Research Report tab.

## The depth bar

Every report follows the same structure (see `GEV.md` for the reference):

1. **The one-paragraph frame** — the central tension that decides how the stock trades
2. **Snapshot** — the decision-relevant facts in one table
3. **Part I — every lever that moves the stock**, ranked, + a recent-move case study
4. **Part II — government, policy & geopolitics**, tailwinds/headwinds by segment
5. **Part III — macro & the global economy**, incl. rates → the multiple
6. **Part IV — is the growth sustainable?** levels vs. derivatives, the bear/cliff case,
   what cushions it, and the valuation synthesis (DCF / WACC / reverse-DCF)
7. **Part V — scenarios** (bull / base / bear with stock implications)
8. **Part VI–VII — dated catalysts + bottom line**
9. **Sources**

> Decision-support only — not investment advice.
