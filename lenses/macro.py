"""Lens 4 — Macro / government.

Two parts:
  • Quantitative rates backdrop from FRED (Fed funds, 2s/10s, CPI) — deterministic.
  • Qualitative policy scan via Claude's web_search tool: current Fed posture plus
    legislation / regulation / tariffs / export controls / subsidies relevant to
    the stock's sector (energy, data centers, AI infrastructure). Each item is
    mapped to tailwind / headwind for THIS ticker, with a cited source and date.

Degrades gracefully: with no Anthropic key it reports the FRED backdrop only.
"""
from __future__ import annotations

import config
from data import fred, llm
from data.prices import MarketData
from models import BEARISH, BULLISH, NA, NEUTRAL, LensResult

_SYSTEM = (
    "You are a rigorous macro/policy analyst. Use web search to find CURRENT, "
    "dated facts. Never fabricate sources. Every signal must cite a real URL and "
    "a date. Be specific to the named company and its sector."
)


def _build_prompt(md: MarketData, rates: dict) -> str:
    company = md.info.get("longName") or md.ticker
    sector = md.sector or "unknown sector"
    theme = config.macro_theme(md.sector)
    rate_line = (
        f"FRED backdrop: Fed funds {rates.get('fed_funds')}%, 10Y {rates.get('ust_10y')}%, "
        f"2Y {rates.get('ust_2y')}%, CPI YoY {rates.get('cpi_yoy')}%."
    )
    return f"""Research the current macro and government/policy backdrop for {company} ({md.ticker}), \
a {sector} company. {rate_line}

Focus on items active in the last ~6 months relevant to: {theme}.
Cover: (1) the current Fed/rates posture and its direction; (2) legislation, \
regulation, tariffs, export controls, subsidies, or permitting changes that \
affect this company's sector.

Use web search. For each distinct item, decide whether it is a TAILWIND, \
HEADWIND, or NEUTRAL for {md.ticker} specifically, and explain why in one line.

Finish your answer with a single JSON code block of this exact shape:
```json
{{
  "rates_posture": "one sentence on Fed/rates direction and what it means for this stock",
  "overall": "tailwind | headwind | mixed",
  "signals": [
    {{"item": "short name", "impact": "tailwind|headwind|neutral", "date": "YYYY-MM or YYYY-MM-DD", "rationale": "one line", "source": "https://..."}}
  ]
}}
```"""


def _verdict_from_signals(signals: list[dict]) -> tuple[str, str]:
    tail = sum(1 for s in signals if str(s.get("impact", "")).lower().startswith("tail"))
    head = sum(1 for s in signals if str(s.get("impact", "")).lower().startswith("head"))
    if tail > head:
        return "Net Tailwind", BULLISH
    if head > tail:
        return "Net Headwind", BEARISH
    return "Balanced", NEUTRAL


def _rates_posture_text(rates: dict) -> str:
    ten, two = rates.get("ust_10y"), rates.get("ust_2y")
    bits = []
    ff = rates.get("fed_funds")
    if ff is not None:
        bits.append(f"Fed funds {ff:.2f}%")
    if ten is not None and two is not None:
        spread = ten - two
        shape = "inverted" if spread < 0 else "normal"
        bits.append(f"2s10s {spread:+.2f}pp ({shape})")
    if rates.get("cpi_yoy") is not None:
        bits.append(f"CPI {rates['cpi_yoy']:.1f}% YoY")
    return ", ".join(bits) if bits else "FRED data unavailable (no key?)"


def run(md: MarketData) -> LensResult:
    rates = fred.rates_snapshot()
    theme = config.macro_theme(md.sector)
    rates_text = _rates_posture_text(rates)

    metrics = {
        "Sector Theme": theme,
        "Fed Funds Rate": _as_frac(rates.get("fed_funds")),
        "US 10Y": _as_frac(rates.get("ust_10y")),
        "US 2Y": _as_frac(rates.get("ust_2y")),
        "Rates Posture": rates_text,
    }

    if not llm.available():
        summary = (
            f"FRED backdrop only (no Anthropic key for the policy web search). "
            f"{rates_text}. Theme to watch: {theme}."
        )
        metrics.update({"Verdict": "Insufficient Data", "Signals": "", "Sources": ""})
        return LensResult("macro", "Insufficient Data", NA, summary, metrics=metrics,
                          raw={"rates": rates}, error="no_llm")

    text = llm.web_research(_build_prompt(md, rates), system=_SYSTEM)
    if not text or text.startswith("__ERROR__"):
        summary = f"Macro web search failed ({text}). FRED backdrop: {rates_text}."
        metrics.update({"Verdict": "Insufficient Data", "Signals": "", "Sources": ""})
        return LensResult("macro", "Insufficient Data", NA, summary, metrics=metrics,
                          raw={"rates": rates, "llm_text": text}, error="llm_failed")

    parsed = llm.extract_json(text) or {}
    signals = parsed.get("signals", []) if isinstance(parsed, dict) else []
    verdict, stance = _verdict_from_signals(signals)
    if not signals:
        verdict, stance = "Balanced", NEUTRAL

    signal_lines = []
    source_lines = []
    for sg in signals:
        impact = str(sg.get("impact", "neutral")).upper()
        signal_lines.append(
            f"[{impact}] {sg.get('item', '?')} ({sg.get('date', 'n/a')}): {sg.get('rationale', '')}"
        )
        if sg.get("source"):
            source_lines.append(f"{sg.get('item', '?')}: {sg['source']}")

    posture = parsed.get("rates_posture", "") if isinstance(parsed, dict) else ""
    summary = (
        f"{verdict}. {posture or rates_text}. "
        + (f"{len(signals)} policy signal(s): "
           + "; ".join(s.split(': ', 1)[0].replace('[', '').replace(']', '')
                       for s in signal_lines[:4])
           if signals else "No specific policy signals surfaced.")
    )

    metrics.update({
        "Rates Posture": posture or rates_text,
        "Verdict": verdict,
        "Signals": "\n".join(signal_lines),
        "Sources": "\n".join(source_lines),
    })
    raw = {"rates": rates, "parsed": parsed, "llm_text": text}
    return LensResult("macro", verdict, stance, summary, metrics=metrics, raw=raw)


def _as_frac(pct_value):
    """FRED returns percent (4.33). Airtable percent fields store fractions."""
    return None if pct_value is None else pct_value / 100.0
