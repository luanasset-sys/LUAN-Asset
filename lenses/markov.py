"""Lens 3 — Markov regime model.

IMPORTANT: this is REGIME FRAMING, not price prediction. It classifies each week
into a discrete state, estimates how states have historically transitioned, and
reports the current state, the next-period distribution, and the long-run steady
state. None of that forecasts price — it describes the *behavioral regime* the
stock has been in. Output text says so explicitly and avoids confidence claims.

Upgrade hook: `classify_states` and `estimate_transition_matrix` are separated so
a Hidden Markov Model (hmmlearn) can replace the threshold classifier later —
see `fit_hmm_states` stub below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data.prices import MarketData
from models import BEARISH, BULLISH, NA, NEUTRAL, LensResult


def _weekly_returns(close: pd.Series) -> pd.Series:
    weekly = close.resample("W-FRI").last().dropna()
    return weekly.pct_change().dropna()


def classify_states(weekly_ret: pd.Series) -> pd.Series:
    """Label each week: '{Up|Flat|Down}/{HiVol|LoVol}'.

    Direction uses fixed return bands (config). The vol tag compares the rolling
    realized vol of recent weeks to the sample median (a stable hi/lo split).
    """
    direction = pd.Series("Flat", index=weekly_ret.index)
    direction[weekly_ret >= config.MARKOV_UP] = "Up"
    direction[weekly_ret <= config.MARKOV_DOWN] = "Down"

    roll_vol = weekly_ret.rolling(config.MARKOV_VOL_WINDOW).std()
    median_vol = roll_vol.median()
    vol_tag = roll_vol.apply(lambda v: "HiVol" if pd.notna(v) and v > median_vol else "LoVol")

    states = direction.str.cat(vol_tag, sep="/")
    return states.dropna()


def estimate_transition_matrix(states: pd.Series) -> tuple[list[str], np.ndarray]:
    """Maximum-likelihood transition matrix from the observed state sequence."""
    labels = sorted(states.unique())
    idx = {s: i for i, s in enumerate(labels)}
    n = len(labels)
    counts = np.zeros((n, n))
    seq = states.tolist()
    for a, b in zip(seq[:-1], seq[1:]):
        counts[idx[a], idx[b]] += 1
    # Row-normalize; rows with no outgoing transitions become uniform.
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.where(row_sums > 0, counts / row_sums, 1.0 / n)
    return labels, mat


def steady_state(mat: np.ndarray, iters: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Stationary distribution via power iteration on the transition matrix."""
    n = mat.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        nxt = pi @ mat
        if np.linalg.norm(nxt - pi, 1) < tol:
            pi = nxt
            break
        pi = nxt
    s = pi.sum()
    return pi / s if s else pi


def fit_hmm_states(weekly_ret: pd.Series, n_states: int = 4):  # pragma: no cover
    """Future upgrade: latent-state HMM via hmmlearn.GaussianHMM.

    Drop-in replacement for `classify_states` — fit on weekly returns (and
    optionally rolling vol as a second feature) and return a decoded state
    sequence. Kept as a stub so the rest of the lens is HMM-ready.
    """
    raise NotImplementedError("Install hmmlearn and implement to upgrade to a latent HMM.")


def _direction_mass(labels: list[str], dist: np.ndarray, key: str) -> float:
    return float(sum(p for s, p in zip(labels, dist) if s.startswith(key)))


def run(md: MarketData) -> LensResult:
    if not md.has_history:
        return LensResult("markov", "Insufficient Data", NA,
                          "Not enough price history for a weekly regime model.",
                          error="no_history")

    weekly_ret = _weekly_returns(md.close)
    if len(weekly_ret) < config.MARKOV_MIN_WEEKS:
        return LensResult("markov", "Insufficient Data", NA,
                          f"Only {len(weekly_ret)} weeks of data; need "
                          f"{config.MARKOV_MIN_WEEKS}+ for a stable matrix.",
                          error="too_few_weeks")

    states = classify_states(weekly_ret)
    labels, mat = estimate_transition_matrix(states)
    current = states.iloc[-1]
    cur_idx = labels.index(current)
    next_dist = mat[cur_idx]
    ss = steady_state(mat)

    up_ss = _direction_mass(labels, ss, "Up")
    down_ss = _direction_mass(labels, ss, "Down")
    up_next = _direction_mass(labels, next_dist, "Up")
    down_next = _direction_mass(labels, next_dist, "Down")

    # Soft stance from the regime tilt — deliberately gentle, clearly labeled.
    margin = 0.06
    if up_ss - down_ss > margin:
        verdict, stance = f"Constructive regime ({current})", BULLISH
    elif down_ss - up_ss > margin:
        verdict, stance = f"Defensive regime ({current})", BEARISH
    else:
        verdict, stance = f"Balanced regime ({current})", NEUTRAL

    def fmt(d):
        return ", ".join(f"{s} {p*100:.0f}%" for s, p in sorted(zip(labels, d), key=lambda x: -x[1]) if p > 0.01)

    summary = (
        f"REGIME FRAMING (not a forecast). Current weekly state: {current}. "
        f"From here, next-week distribution leans Up {up_next*100:.0f}% / "
        f"Down {down_next*100:.0f}%. Long-run steady state: Up {up_ss*100:.0f}% / "
        f"Down {down_ss*100:.0f}% across {len(weekly_ret)} weeks. "
        f"This describes the behavioral regime, not where the price is going."
    )

    metrics = {
        "Current State": current,
        "States": ", ".join(labels),
        "Weeks Analyzed": len(weekly_ret),
        "Transition Matrix": _matrix_text(labels, mat),
        "Next Period Dist": fmt(next_dist),
        "Steady State": fmt(ss),
        "Note": "Regime framing, not price prediction. HMM-upgrade hook in place.",
    }
    raw = {
        "labels": labels,
        "transition_matrix": mat.round(4).tolist(),
        "next_period": dict(zip(labels, next_dist.round(4).tolist())),
        "steady_state": dict(zip(labels, ss.round(4).tolist())),
        "current_state": current,
        "weeks": len(weekly_ret),
    }
    return LensResult("markov", verdict, stance, summary, metrics=metrics, raw=raw)


def _matrix_text(labels: list[str], mat: np.ndarray) -> str:
    lines = ["from\\to | " + " ".join(f"{l:>11}" for l in labels)]
    for i, l in enumerate(labels):
        row = " ".join(f"{mat[i, j]*100:>10.0f}%" for j in range(len(labels)))
        lines.append(f"{l:>11} | {row}")
    return "\n".join(lines)
