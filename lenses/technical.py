"""Lens 2 — Quant / technical. Numbers only, no narrative judgment about *why*.

Reads daily prices and answers "what path is it on": trend (50/200 MAs and the
golden/death cross), momentum (RSI, 3/6/12-mo returns), risk (realized vol, max
drawdown), and relative strength vs SPY and the sector ETF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.prices import MarketData
from models import BEARISH, BULLISH, NA, NEUTRAL, LensResult

_TRADING_DAYS = {"3M": 63, "6M": 126, "12M": 252}


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def _trailing_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    past = close.iloc[-(days + 1)]
    if past == 0:
        return None
    return float(close.iloc[-1] / past - 1.0)


def _realized_vol(close: pd.Series, days: int = 252) -> float | None:
    rets = np.log(close / close.shift(1)).dropna().iloc[-days:]
    if len(rets) < 20:
        return None
    return float(rets.std() * np.sqrt(252))


def _max_drawdown(close: pd.Series, days: int = 252) -> float | None:
    window = close.iloc[-days:]
    if len(window) < 20:
        return None
    running_max = window.cummax()
    dd = window / running_max - 1.0
    return float(dd.min())


def _cross_state(ma50: pd.Series, ma200: pd.Series) -> str:
    if ma50.dropna().empty or ma200.dropna().empty:
        return "Unknown"
    above = ma50.iloc[-1] > ma200.iloc[-1]
    # was the cross recent (within ~20 sessions)?
    recent = (ma50 - ma200).dropna().iloc[-25:]
    crossed_recently = len(recent) > 1 and (np.sign(recent.iloc[0]) != np.sign(recent.iloc[-1]))
    if above:
        return "Golden Cross (recent)" if crossed_recently else "Golden Cross (50>200)"
    return "Death Cross (recent)" if crossed_recently else "Death Cross (50<200)"


def _rel_strength(asset: pd.Series, bench: pd.Series, days: int = 126) -> float | None:
    a = _trailing_return(asset, days)
    b = _trailing_return(bench, days)
    if a is None or b is None:
        return None
    return float((1 + a) / (1 + b) - 1.0)


def run(md: MarketData) -> LensResult:
    if not md.has_history:
        return LensResult("technical", "Insufficient Data", NA,
                          "Not enough price history from yfinance.", error="no_history")

    close = md.close
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    cross = _cross_state(ma50, ma200)
    rsi = _rsi(close)
    rets = {k: _trailing_return(close, d) for k, d in _TRADING_DAYS.items()}
    vol = _realized_vol(close)
    mdd = _max_drawdown(close)
    rs_spy = _rel_strength(close, md.spy_close)
    rs_sector = (None if md.sector_etf == "SPY"
                 else _rel_strength(close, md.sector_close))

    # ── Scoring: aggregate signed signals into a -3..+3 trend score ───────────
    score = 0
    if "Golden" in cross:
        score += 1 + (1 if "recent" in cross else 0)
    elif "Death" in cross:
        score -= 1 + (1 if "recent" in cross else 0)
    if rets["6M"] is not None:
        score += 1 if rets["6M"] > 0 else -1
    if rets["12M"] is not None:
        score += 1 if rets["12M"] > 0 else -1
    if rs_spy is not None:
        score += 1 if rs_spy > 0 else -1

    if score >= 3:
        verdict, stance = "Uptrend", BULLISH
    elif score == 2:
        verdict, stance = "Improving", BULLISH
    elif score <= -3:
        verdict, stance = "Downtrend", BEARISH
    elif score == -2:
        verdict, stance = "Deteriorating", BEARISH
    else:
        verdict, stance = "Neutral", NEUTRAL

    rsi_note = ""
    if rsi is not None:
        if rsi >= 70:
            rsi_note = f" RSI {rsi:.0f} (overbought)."
        elif rsi <= 30:
            rsi_note = f" RSI {rsi:.0f} (oversold)."
        else:
            rsi_note = f" RSI {rsi:.0f}."

    def pct(x):
        return "n/a" if x is None else f"{x * 100:+.1f}%"

    summary = (
        f"{verdict}. {cross}; 3/6/12-mo returns {pct(rets['3M'])} / {pct(rets['6M'])} / "
        f"{pct(rets['12M'])}.{rsi_note} Realized vol "
        f"{'n/a' if vol is None else f'{vol*100:.0f}%'}, max drawdown (1y) {pct(mdd)}. "
        f"Relative strength vs SPY {pct(rs_spy)}"
        + (f", vs {md.sector_etf} {pct(rs_sector)}." if rs_sector is not None else ".")
    )

    metrics = {
        "Price": md.price,
        "MA50": float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else None,
        "MA200": float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else None,
        "Cross State": cross,
        "RSI14": rsi,
        "Return 3M": rets["3M"],
        "Return 6M": rets["6M"],
        "Return 12M": rets["12M"],
        "Realized Vol": vol,
        "Max Drawdown": mdd,
        "RS vs SPY": rs_spy,
        "RS vs Sector": rs_sector,
        "Sector ETF": md.sector_etf,
    }
    return LensResult("technical", verdict, stance, summary, metrics=metrics,
                      raw={"score": score, **metrics})
