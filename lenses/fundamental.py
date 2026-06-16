"""Lens 1 — Fundamental (hedge-fund style).

Builds a simple model from SEC EDGAR annual financials and yfinance market data:
  • Valuation  — DCF (bull/base/bear) + P/E, EV/EBITDA, EV/Sales vs the stock's
                 OWN 5-year multiple history (percentile rank).
  • Quality    — margins, ROIC, FCF conversion, net debt / EBITDA.
  • Growth     — revenue & EPS CAGR.
Outputs a fair-value range, a 5-point verdict, and plain-English reasoning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data import edgar
from data.prices import MarketData
from models import (BEARISH, BULLISH, NA, NEUTRAL, LensResult)

_TAX = 0.21  # flat NOPAT tax assumption for ROIC


def _latest(series: dict[int, float], years: list[int]) -> float | None:
    for y in reversed(years):
        if y in series:
            return series[y]
    return None


def _cagr(series: dict[int, float], years: list[int], span: int = 3) -> float | None:
    avail = [y for y in years if y in series]
    if len(avail) < 2:
        return None
    end_y = avail[-1]
    start_y = next((y for y in avail if y <= end_y - span), avail[0])
    start, end = series.get(start_y), series.get(end_y)
    n = end_y - start_y
    if not start or not end or n <= 0 or start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / n) - 1.0


def _price_at_year_end(close: pd.Series, year: int) -> float | None:
    if close is None or close.empty:
        return None
    sub = close[close.index.year == year]
    return float(sub.iloc[-1]) if len(sub) else None


def _percentile(value: float | None, history: list[float]) -> float | None:
    """Fraction of historical values <= current (incl current). Lower = cheaper."""
    pts = [h for h in history if h is not None and np.isfinite(h)]
    if value is None or not np.isfinite(value) or not pts:
        return None
    pts = pts + [value]
    return sum(1 for h in pts if h <= value) / len(pts)


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _dcf_fair_value(base_fcf, growth, net_debt, shares):
    if not base_fcf or not shares or shares <= 0 or base_fcf <= 0:
        return None
    r, tg = config.DCF_DISCOUNT_RATE, config.DCF_TERMINAL_GROWTH
    g = max(min(growth if growth is not None else 0.05, 0.20), -0.05)
    pv = 0.0
    fcf = base_fcf
    for yr in range(1, config.DCF_YEARS + 1):
        fcf *= (1 + g)
        pv += fcf / (1 + r) ** yr
    terminal = fcf * (1 + tg) / (r - tg)
    pv += terminal / (1 + r) ** config.DCF_YEARS
    equity = pv - (net_debt or 0)
    return equity / shares


def run(md: MarketData) -> LensResult:
    data = edgar.fundamentals_annual(md.ticker)
    if "error" in data:
        return LensResult("fundamental", "Insufficient Data", NA,
                          data["error"], error="no_edgar")

    s = data["series"]
    years = data["fiscal_years"]
    if not years:
        return LensResult("fundamental", "Insufficient Data", NA,
                          "No annual financials parsed from EDGAR.", error="no_years")

    rev = _latest(s["revenue"], years)
    ni = _latest(s["net_income"], years)
    gp = _latest(s["gross_profit"], years)
    oi = _latest(s["operating_income"], years)
    da = _latest(s["depreciation_amortization"], years)
    ocf = _latest(s["operating_cash_flow"], years)
    capex = _latest(s["capex"], years)
    eps = _latest(s["eps_diluted"], years)
    debt = (_latest(s["total_debt"], years) or 0) + (_latest(s["current_debt"], years) or 0)
    cash = _latest(s["cash"], years) or 0
    equity = _latest(s["equity"], years)

    net_debt = debt - cash
    ebitda = (oi + da) if (oi is not None and da is not None) else oi
    fcf = (ocf - capex) if (ocf is not None and capex is not None) else None

    # ── Quality ──────────────────────────────────────────────────────────────
    gross_margin = _safe_div(gp, rev)
    op_margin = _safe_div(oi, rev)
    net_margin = _safe_div(ni, rev)
    fcf_margin = _safe_div(fcf, rev)
    fcf_conversion = _safe_div(fcf, ni)
    nopat = oi * (1 - _TAX) if oi is not None else None
    invested_capital = (debt + (equity or 0) - cash) if equity is not None else None
    roic = _safe_div(nopat, invested_capital)
    nd_ebitda = _safe_div(net_debt, ebitda)

    # ── Growth ───────────────────────────────────────────────────────────────
    rev_cagr = _cagr(s["revenue"], years)
    eps_cagr = _cagr(s["eps_diluted"], years)

    # ── Valuation: current multiples ─────────────────────────────────────────
    price, mcap, shares = md.price, md.market_cap, md.shares_outstanding
    if mcap is None and price and shares:
        mcap = price * shares
    ev = (mcap + net_debt) if mcap is not None else None
    pe = _safe_div(price, eps) if (eps and eps > 0) else None
    ev_ebitda = _safe_div(ev, ebitda) if (ebitda and ebitda > 0) else None
    ev_sales = _safe_div(ev, rev)

    # ── Valuation: own 5y multiple history → percentile ──────────────────────
    pe_hist, ev_ebitda_hist, ev_sales_hist = [], [], []
    for y in years[-5:]:
        py = _price_at_year_end(md.close, y)
        eps_y = s["eps_diluted"].get(y)
        ni_y = s["net_income"].get(y)
        rev_y = s["revenue"].get(y)
        oi_y = s["operating_income"].get(y)
        da_y = s["depreciation_amortization"].get(y)
        ebitda_y = (oi_y + da_y) if (oi_y is not None and da_y is not None) else oi_y
        nd_y = ((s["total_debt"].get(y, 0) or 0) + (s["current_debt"].get(y, 0) or 0)
                - (s["cash"].get(y, 0) or 0))
        shares_y = _safe_div(ni_y, eps_y) if (eps_y and eps_y > 0) else shares
        if py and eps_y and eps_y > 0:
            pe_hist.append(py / eps_y)
        if py and shares_y and ebitda_y and ebitda_y > 0:
            ev_y = py * shares_y + nd_y
            ev_ebitda_hist.append(ev_y / ebitda_y)
            if rev_y:
                ev_sales_hist.append(ev_y / rev_y)

    pe_pct = _percentile(pe, pe_hist)
    ev_ebitda_pct = _percentile(ev_ebitda, ev_ebitda_hist)
    ev_sales_pct = _percentile(ev_sales, ev_sales_hist)

    # ── DCF fair value: bull / base / bear ───────────────────────────────────
    fair = {}
    for name, factor in config.DCF_SCENARIOS.items():
        g = (rev_cagr if rev_cagr is not None else 0.05) * factor
        fair[name] = _dcf_fair_value(fcf, g, net_debt, shares)
    fv_bear, fv_base, fv_bull = fair.get("bear"), fair.get("base"), fair.get("bull")

    # ── Scoring ──────────────────────────────────────────────────────────────
    score = 0
    notes: list[str] = []
    if price and fv_base:
        if fv_bear and price < fv_bear:
            score += 2; notes.append("price below bear-case DCF (cheap)")
        elif price < fv_base:
            score += 1; notes.append("price below base-case DCF")
        elif fv_bull and price > fv_bull:
            score -= 2; notes.append("price above bull-case DCF (rich)")
        elif price > fv_base:
            score -= 1; notes.append("price above base-case DCF")
    for pct, label in ((pe_pct, "P/E"), (ev_ebitda_pct, "EV/EBITDA"), (ev_sales_pct, "EV/Sales")):
        if pct is not None:
            if pct <= 0.40:
                score += 1; notes.append(f"{label} cheap vs own 5y ({pct*100:.0f}th pct)")
            elif pct >= 0.75:
                score -= 1; notes.append(f"{label} rich vs own 5y ({pct*100:.0f}th pct)")
    if roic is not None:
        if roic > 0.15:
            score += 1; notes.append(f"high ROIC {roic*100:.0f}%")
        elif roic < 0.05:
            score -= 1; notes.append(f"low ROIC {roic*100:.0f}%")
    if nd_ebitda is not None:
        if nd_ebitda < 1:
            score += 1; notes.append("low leverage")
        elif nd_ebitda > 4:
            score -= 1; notes.append(f"high leverage ({nd_ebitda:.1f}x)")
    if fcf_margin is not None and fcf_margin > 0.15:
        score += 1; notes.append(f"strong FCF margin {fcf_margin*100:.0f}%")
    if rev_cagr is not None:
        if rev_cagr > 0.10:
            score += 1; notes.append(f"revenue CAGR {rev_cagr*100:.0f}%")
        elif rev_cagr < 0:
            score -= 1; notes.append("shrinking revenue")

    if score >= 3:
        verdict, stance = "Bullish", BULLISH
    elif score in (1, 2):
        verdict, stance = "Lean Bullish", BULLISH
    elif score == 0:
        verdict, stance = "Neutral", NEUTRAL
    elif score in (-1, -2):
        verdict, stance = "Lean Bearish", BEARISH
    else:
        verdict, stance = "Bearish", BEARISH

    def money(x):
        return "n/a" if x is None else f"${x:,.0f}"

    def mult(x):
        return "n/a" if x is None else f"{x:.1f}x"

    def pct_(x):
        return "n/a" if x is None else f"{x*100:.1f}%"

    fv_str = (f"${fv_bear:,.0f} / ${fv_base:,.0f} / ${fv_bull:,.0f}"
              if all(v is not None for v in (fv_bear, fv_base, fv_bull)) else "n/a")
    summary = (
        f"{verdict}. Fair-value range (bear/base/bull DCF): {fv_str} vs price "
        f"${price:,.2f}. " if price else f"{verdict}. Fair-value range: {fv_str}. "
    )
    summary += (
        f"Valuation: P/E {mult(pe)} ({pct_(pe_pct)} of own 5y), "
        f"EV/EBITDA {mult(ev_ebitda)} ({pct_(ev_ebitda_pct)}), "
        f"EV/Sales {mult(ev_sales)} ({pct_(ev_sales_pct)}). "
        f"Quality: ROIC {pct_(roic)}, op margin {pct_(op_margin)}, FCF margin "
        f"{pct_(fcf_margin)}, FCF conversion {pct_(fcf_conversion)}, net debt/EBITDA "
        f"{mult(nd_ebitda)}. Growth: revenue CAGR {pct_(rev_cagr)}, EPS CAGR {pct_(eps_cagr)}. "
        f"Drivers: {'; '.join(notes) if notes else 'mixed signals'}."
    )

    metrics = {
        "Revenue TTM": rev,
        "Revenue CAGR 3y": rev_cagr,
        "EPS CAGR 3y": eps_cagr,
        "Gross Margin": gross_margin,
        "Operating Margin": op_margin,
        "Net Margin": net_margin,
        "FCF Margin": fcf_margin,
        "ROIC": roic,
        "FCF Conversion": fcf_conversion,
        "Net Debt to EBITDA": nd_ebitda,
        "PE": pe,
        "EV EBITDA": ev_ebitda,
        "EV Sales": ev_sales,
        "PE 5y Pctile": pe_pct,
        "EV EBITDA 5y Pctile": ev_ebitda_pct,
        "EV Sales 5y Pctile": ev_sales_pct,
        "DCF Fair Value": fv_base,
        "Bull FV": fv_bull,
        "Base FV": fv_base,
        "Bear FV": fv_bear,
    }
    raw = {
        "entity": data.get("entity"),
        "cik": data.get("cik"),
        "fiscal_years": years,
        "latest": {"revenue": rev, "net_income": ni, "ebitda": ebitda, "fcf": fcf,
                   "net_debt": net_debt, "shares": shares, "eps": eps},
        "score": score,
        "notes": notes,
        **{k: v for k, v in metrics.items()},
    }
    return LensResult("fundamental", verdict, stance, summary, metrics=metrics, raw=raw)
