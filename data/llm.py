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
