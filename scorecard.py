"""Assemble the four lenses into a scorecard.

Deliberately does NOT collapse to a single buy/sell number. It shows the four
verdicts side by side and flags where they AGREE vs DISAGREE — the disagreement
is the signal.
"""
from __future__ import annotations

from models import BEARISH, BULLISH, NA, NEUTRAL, LensResult, RunResult

_STANCE_LABEL = {BULLISH: "▲ bullish", BEARISH: "▼ bearish", NEUTRAL: "● neutral", NA: "– n/a"}


def _agreement(results: list[LensResult]) -> tuple[str, bool]:
    rated = [r for r in results if r.stance != NA]
    bulls = [r.lens for r in rated if r.stance == BULLISH]
    bears = [r.lens for r in rated if r.stance == BEARISH]
    neutrals = [r.lens for r in rated if r.stance == NEUTRAL]

    disagreement = bool(bulls and bears)
    lines = []
    for r in results:
        lines.append(f"  {r.lens:<12} {_STANCE_LABEL[r.stance]:<12} {r.verdict}")

    verdict_line = ""
    if not rated:
        verdict_line = "No lens produced a rated view (insufficient data)."
    elif disagreement:
        verdict_line = (
            f"DISAGREEMENT (this is the signal): bullish [{', '.join(bulls)}] "
            f"vs bearish [{', '.join(bears)}]"
            + (f"; neutral [{', '.join(neutrals)}]" if neutrals else "")
            + ". Dig into why the lenses conflict before acting."
        )
    elif bulls and not bears:
        verdict_line = (
            f"ALIGNED BULLISH: {', '.join(bulls)} agree"
            + (f"; {', '.join(neutrals)} neutral" if neutrals else "") + "."
        )
    elif bears and not bulls:
        verdict_line = (
            f"ALIGNED BEARISH: {', '.join(bears)} agree"
            + (f"; {', '.join(neutrals)} neutral" if neutrals else "") + "."
        )
    else:
        verdict_line = "Mostly neutral across lenses; no strong cross-lens signal."

    return "\n".join([verdict_line, "", *lines]), disagreement


def assemble(ticker: str, run_id: str, timestamp: str, price, fundamental, technical,
             markov, macro) -> RunResult:
    agreement_map, disagreement = _agreement([fundamental, technical, markov, macro])
    fair_value = {
        "low": fundamental.metrics.get("Bear FV"),
        "base": fundamental.metrics.get("Base FV"),
        "high": fundamental.metrics.get("Bull FV"),
    }
    return RunResult(
        ticker=ticker, run_id=run_id, timestamp=timestamp, price=price,
        fundamental=fundamental, technical=technical, markov=markov, macro=macro,
        agreement_map=agreement_map, disagreement=disagreement, fair_value=fair_value,
    )


def render(run: RunResult, disclaimer: str) -> str:
    """The printed one-page scorecard."""
    bar = "═" * 74
    fv = run.fair_value
    fv_line = ""
    if all(fv.get(k) is not None for k in ("low", "base", "high")):
        fv_line = (f"Fair value (bear/base/bull): ${fv['low']:,.0f} / ${fv['base']:,.0f} "
                   f"/ ${fv['high']:,.0f}")
    price_line = f"Price: ${run.price:,.2f}" if run.price else "Price: n/a"

    out = [
        bar,
        f" EQUITY RESEARCH HQ  —  {run.ticker}",
        f" {run.timestamp}   {price_line}" + (f"   |   {fv_line}" if fv_line else ""),
        bar,
        "",
        " FOUR LENSES",
        f"  1. Fundamental  [{run.fundamental.verdict}]",
        f"       {_wrap(run.fundamental.summary)}",
        f"  2. Technical    [{run.technical.verdict}]",
        f"       {_wrap(run.technical.summary)}",
        f"  3. Markov       [{run.markov.verdict}]",
        f"       {_wrap(run.markov.summary)}",
        f"  4. Macro/Gov    [{run.macro.verdict}]",
        f"       {_wrap(run.macro.summary)}",
        "",
        " ─ AGREEMENT MAP " + "─" * 57,
        _indent(run.agreement_map),
        "",
        bar,
        f" {disclaimer}",
        bar,
    ]
    return "\n".join(out)


def _wrap(text: str, width: int = 96) -> str:
    import textwrap
    return "\n       ".join(textwrap.wrap(text, width)) if text else "(none)"


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())
