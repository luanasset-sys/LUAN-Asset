"""Anthropic wrapper: live web research (macro lens) + optional narrative polish.

Uses claude-opus-4-8 with the server-side web_search tool. All functions return
None / pass-through text when no API key is configured, so the pipeline still
runs free (verdicts fall back to deterministic templates).
"""
from __future__ import annotations

import json
import re
from typing import Any

import config

try:
    import anthropic
except ImportError:  # keeps the module importable even if SDK missing
    anthropic = None  # type: ignore

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
_MAX_CONTINUATIONS = 6


def available() -> bool:
    return bool(anthropic and config.ANTHROPIC_API_KEY)


def _client():
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _collect_text(message) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def web_research(prompt: str, system: str | None = None, max_tokens: int = 4000) -> str | None:
    """Run a web-search-enabled turn and return the model's final text.

    Handles the server-side tool loop: when the model pauses (`pause_turn`)
    mid-search, we re-send the conversation so it resumes automatically.
    """
    if not available():
        return None
    client = _client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        for _ in range(_MAX_CONTINUATIONS):
            resp = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system or "You are a rigorous macro/policy research analyst.",
                tools=[WEB_SEARCH_TOOL],
                messages=messages,
            )
            if resp.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            return _collect_text(resp)
    except Exception as exc:  # network / auth / model errors shouldn't kill the run
        return f"__ERROR__ {exc}"
    return None


def synthesize(prompt: str, system: str | None = None, max_tokens: int = 600) -> str | None:
    """Plain completion used to rephrase computed metrics into natural prose."""
    if not available():
        return None
    try:
        resp = _client().messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system or "You write tight, plain-English equity research notes.",
            messages=[{"role": "user", "content": prompt}],
        )
        return _collect_text(resp).strip()
    except Exception:
        return None


# ── Deep-dive report pipeline ────────────────────────────────────────────────
# A single LLM call can't produce an institutional-grade deep dive. This runs a
# multi-pass pipeline: two web-research dossiers (company/financials + policy/
# macro), then section-by-section synthesis grounded in those dossiers AND the
# computed model (DCF / WACC / reverse-DCF / multiples). Mirrors the structure of
# a sell-side deep dive: frame, every lever, policy, macro, the levels-vs-
# derivatives sustainability debate, scenarios, dated catalysts, bottom line.

_REPORT_SYSTEM = (
    "You are a buy-side analyst writing an institutional-grade equity deep dive. "
    "Write in precise, concrete, plain English: specific numbers, named policies, "
    "dated catalysts, real cause-and-effect mechanisms. No filler, no boilerplate, "
    "no hedging clichés. Rigorously distinguish a LEVEL from its rate of change "
    "(first/second derivative). Take the bear case seriously. Cite sources inline "
    "with dates when you rely on a specific external fact. Output GitHub-flavored "
    "markdown. Ground every figure in the supplied dossier and model data — never "
    "invent numbers; if something is unknown, say so."
)


def _report_sections(head: str):
    """(progress label, max_tokens, instruction) for each section of the deep dive."""
    return [
        ("Frame & snapshot", 2200,
         f"Write the TOP of the report. Start with a level-1 header `# {head} — Deep Dive`, "
         f"then an italic line with the as-of date and 'Decision-support, not investment "
         f"advice.' Then `## The one-paragraph frame`: a single tight paragraph naming the "
         f"central tension that determines how this stock actually trades. Then `## Snapshot`: "
         f"a two-column markdown table of the most decision-relevant facts — price, market cap, "
         f"52-week range, latest guidance/forward estimates, the key leading-indicator level, "
         f"segment mix, valuation multiples, the model's bottom-up WACC and DCF fair-value "
         f"range, consensus rating & price-target range, and the most recent analyst moves."),
        ("Part I — every lever that moves the stock", 4800,
         "Write `## Part I — Why the stock moves: every lever`. Enumerate the 8–12 DISTINCT "
         "drivers that actually move THIS stock, ranked by current importance, each as a "
         "numbered **bold** item followed by a substantive paragraph: what the driver is, its "
         "current reading, the precise mechanism by which it moves the price, and what would "
         "change it. Be specific to this company's economics — not generic. If the dossier "
         "shows a notable recent share-price move, end with a bold 'Recent move — case study' "
         "paragraph dissecting what actually caused it and what it reveals about how the stock "
         "is priced."),
        ("Part II — government & policy", 4200,
         "Write `## Part II — Government, policy & geopolitics`. One opening line on why "
         "government is a first-order (not background) input for this company. Then "
         "`### Tailwinds` and `### Headwinds` with specific, named, dated policies, "
         "regulations, executive orders, subsidies/tax-credits, tariffs and court actions, "
         "organized by segment or business line, each tracing the transmission to revenue, "
         "margins or orders. Then `### Emerging / swing risks` for policy that could break "
         "either way. End with a one-line **Net policy read**. Cite sources with dates."),
        ("Part III — macro & global economy", 4000,
         "Write `## Part III — Global economy & macro`. Cover, as sub-sections: "
         "`### Rates & the multiple` — the current rate setting and path, and EXPLICITLY how a "
         "higher/lower-for-longer discount rate re-rates a long-duration equity like this "
         "(tie to the model's WACC); `### The secular cycle` — the multi-year demand force the "
         "company rides, with a growth rate; `### Global demand breadth` — regional drivers and "
         "diversification; `### The macro risk that rhymes with the thesis` — the main "
         "macro/sentiment risk and why it hits this name. Specific numbers and sources."),
        ("Part IV — sustainability & valuation (the core)", 5200,
         "Write `## Part IV — Is the growth sustainable, and at what derivative?`. This is the "
         "heart of the report — be rigorous and the longest here. Sub-sections: "
         "`### Levels vs. derivatives` — present the trajectory of the key leading indicator "
         "over recent quarters (a small markdown table if data allows) and analyze the LEVEL, "
         "the FIRST derivative (growth rate) and the SECOND derivative (is growth itself "
         "accelerating or decelerating?), calling out comp effects and one-offs; "
         "`### The forward numbers` — what is high-confidence/visible vs contested, using "
         "guidance; `### The bear case` — take the strongest saturation / cliff / competition "
         "argument seriously and concretely; `### What cushions it` — the genuine offsets "
         "(recurring revenue, adjacent longer cycles, optionality), each explained; "
         "`### Synthesis & valuation` — your honest read separating the company's path from the "
         "stock's path, and weave in the VALUATION explicitly: the model's DCF fair-value "
         "range, the bottom-up WACC, multiples vs the company's own history, and the "
         "reverse-DCF (what growth the current price implies vs what is realistic). Name the "
         "central asymmetry."),
        ("Part V — scenarios", 2600,
         "Write `## Part V — Scenarios`. A markdown table with rows Bull / Base / Bear "
         "(re-acceleration / normalization / cliff-or-digestion) and columns: What happens · "
         "Key-variable behavior · Stock implication (tie to a price range using the consensus "
         "price-target range and the model fair value). After the table, one or two sentences "
         "naming the most probable scenario and what the current price seems least prepared "
         "for."),
        ("Part VI–VII — catalysts & bottom line", 3200,
         "Write `## Part VI — What to watch`: a dated, specific catalyst checklist (next "
         "earnings date, product/policy decisions, and the metrics whose change would confirm "
         "or break the thesis). Then `## Part VII — Bottom line`: a sharp synthesis of business "
         "quality vs. price — what's settled, what's still debated, the fair-value range and "
         "exactly what must be true to justify today's price, and the single central risk to "
         "underwrite. End with this exact line on its own: "
         "`*Decision-support only — not investment advice.*`"),
    ]


def deep_dive_report(meta: dict, model_ctx: str, progress=None) -> str | None:
    """Run the full multi-pass deep-dive pipeline. Returns assembled markdown.

    `meta`: {company, ticker, sector, asof}. `model_ctx`: the computed-model
    grounding string. `progress`: optional callback(str) for live UI status.
    """
    if not available():
        return None
    company = meta.get("company") or meta.get("ticker", "")
    ticker = meta.get("ticker", "")
    sector = meta.get("sector", "")
    asof = meta.get("asof", "")
    head = f"{company} ({ticker})" if ticker else company

    def note(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    note("Researching financials, segments, analysts & recent price action…")
    dossier_fin = web_research(
        f"You are building an internal research dossier on {head} as of {asof}. Using web "
        f"search, gather CURRENT specifics with dates and source URLs: latest quarterly "
        f"results and forward guidance; revenue / EPS / margin / free-cash-flow trajectory; "
        f"THE key leading indicator for this business and its quarter-by-quarter path over the "
        f"last 5–6 quarters (orders, backlog/RPO, bookings, subscribers, units, same-store — "
        f"whatever actually drives it), with figures; segment breakdown and per-segment growth; "
        f"capacity or operational constraints; capital returns (buyback/dividend); recent "
        f"analyst rating and price-target changes with the consensus range; insider activity; "
        f"and any notable share-price move in the last few months with the SPECIFIC reason. Be "
        f"dense and factual — bullet points with numbers, dates and URLs.",
        system="You are a meticulous equity research analyst gathering primary-source facts.",
        max_tokens=7000) or "(financial dossier unavailable)"

    note("Researching policy, regulation, geopolitics & macro…")
    dossier_macro = web_research(
        f"Build an internal policy & macro dossier on {head} (sector: {sector or 'n/a'}) as of "
        f"{asof}. Using web search, gather with dates and URLs: the specific government "
        f"policies, regulations, executive orders, subsidies / tax-credits, tariffs and "
        f"legal/court actions that help or hurt this company, BY SEGMENT; relevant "
        f"geopolitical factors; the current central-bank rate setting and rate path and what a "
        f"long-duration equity's multiple does in that regime; the secular demand cycle this "
        f"company rides and its growth rate; global/regional demand breadth; and the single "
        f"biggest macro risk to the thesis. Dense bullets with numbers, dates and URLs.",
        system="You are a meticulous policy and macro research analyst gathering sourced facts.",
        max_tokens=7000) or "(policy/macro dossier unavailable)"

    base_ctx = (
        f"COMPUTED MODEL DATA (authoritative — do not contradict these figures):\n{model_ctx}\n\n"
        f"=== FINANCIAL / COMPANY DOSSIER ===\n{dossier_fin}\n\n"
        f"=== POLICY / MACRO DOSSIER ===\n{dossier_macro}\n\n"
    )

    parts: list[str] = []
    sections = _report_sections(head)
    for i, (label, mt, instruction) in enumerate(sections, 1):
        note(f"Writing section {i}/{len(sections) + 1}: {label}…")
        text = synthesize(base_ctx + "TASK:\n" + instruction,
                          system=_REPORT_SYSTEM, max_tokens=mt)
        parts.append(text.strip() if text else f"_({label} unavailable — try regenerating.)_")

    note(f"Writing section {len(sections) + 1}/{len(sections) + 1}: Sources…")
    src = synthesize(
        f"From this research dossier, compile a clean, de-duplicated `## Sources` list of the "
        f"key sources actually referenced, grouped under bold headings (Company filings & "
        f"calls · Analyst & market · Policy & regulation · Macro). Prefer primary filings; "
        f"include publication/access dates and markdown links where a URL is present. Output "
        f"only the markdown.\n\nDOSSIER:\n{dossier_fin}\n\n{dossier_macro}",
        system=_REPORT_SYSTEM, max_tokens=1800)
    if src:
        parts.append(src.strip())

    return "\n\n".join(parts)


def extract_json(text: str) -> Any | None:
    """Pull the last ```json ...``` block (or a bare {...}) out of model text."""
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    candidates = list(fenced)
    if not candidates:
        # fall back to the largest brace-balanced span
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            candidates = [m.group(1)]
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None
