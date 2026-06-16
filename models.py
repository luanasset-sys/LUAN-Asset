"""Typed result objects shared across lenses, scorecard, and sinks."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# Normalized directional stance used for the agree/disagree logic.
BULLISH = "bullish"
NEUTRAL = "neutral"
BEARISH = "bearish"
NA = "na"

STANCE_SCORE = {BULLISH: 1, NEUTRAL: 0, BEARISH: -1, NA: None}


@dataclass
class LensResult:
    """One lens's output. `metrics` feeds the detail table; `raw` is the
    full snapshot persisted for backtesting."""

    lens: str                      # "fundamental" | "technical" | "markov" | "macro"
    verdict: str                   # human-facing verdict label
    stance: str = NA               # normalized: bullish | neutral | bearish | na
    summary: str = ""              # plain-English reasoning
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RunResult:
    """Everything produced for one `python hq.py TICKER` invocation."""

    ticker: str
    run_id: str
    timestamp: str                 # ISO-8601
    price: float | None
    fundamental: LensResult
    technical: LensResult
    markov: LensResult
    macro: LensResult
    agreement_map: str = ""
    disagreement: bool = False
    fair_value: dict[str, float | None] = field(default_factory=dict)

    @property
    def lenses(self) -> list[LensResult]:
        return [self.fundamental, self.technical, self.markov, self.macro]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "price": self.price,
            "agreement_map": self.agreement_map,
            "disagreement": self.disagreement,
            "fair_value": self.fair_value,
            "fundamental": self.fundamental.as_dict(),
            "technical": self.technical.as_dict(),
            "markov": self.markov.as_dict(),
            "macro": self.macro.as_dict(),
        }
