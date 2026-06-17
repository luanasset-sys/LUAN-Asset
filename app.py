#!/usr/bin/env python3
"""Equity Research HQ — interactive dashboard.

    streamlit run app.py

A browser UI over the same four lenses as `hq.py`, plus an interactive DCF you
can drive with sliders (growth, discount rate, terminal growth, horizon) and a
live sensitivity grid, the full per-year financials/ratios, valuation-vs-history
charts, and a side-by-side comparison view.

Decision-support only — not financial advice.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

import config
import scorecard as sc
from data.prices import fetch_market_data
from lenses import fundamental, macro, markov, technical

st.set_page_config(page_title="Equity Research HQ", page_icon="📊", layout="wide")

# ── formatting helpers ───────────────────────────────────────────────────────
def money(x, dp=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.{dp}f}"


def pct(x, dp=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{dp}f}%"


def mult(x, dp=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}x"


def big(x):
    """Human-readable large dollar figure: $1.2B, $345M."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"${x/div:,.1f}{suf}"
    return f"${x:,.0f}"


STANCE_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡", "na": "⚪"}


def _na_macro():
    from models import NA, LensResult
    return LensResult("macro", "Not run", NA, "Enable the macro lens in the sidebar.")


# ── cached data loaders (so slider moves don't refetch the network) ──────────
@st.cache_data(show_spinner="Fetching prices + SEC filings…", ttl=3600)
def load_core(ticker: str):
    md = fetch_market_data(ticker)
    f = fundamental.run(md)
    t = technical.run(md)
    mk = markov.run(md)
    detail = fundamental.model_detail(md)
    return md, f, t, mk, detail


@st.cache_data(show_spinner="Running macro web search…", ttl=3600)
def load_macro(ticker: str):
    md = fetch_market_data(ticker)
    return macro.run(md)


# ── sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Equity Research HQ")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
run_macro = st.sidebar.checkbox("Run macro web-search lens (uses Anthropic key)", value=False)
st.sidebar.caption(config.DISCLAIMER)

if not ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

try:
    md, f, t, mk, detail = load_core(ticker)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load {ticker}: {exc}")
    st.stop()

if md.price is None and not md.has_history:
    st.error(f"No market data for {ticker}. Check the symbol.")
    st.stop()

mc = load_macro(ticker) if run_macro else None
company = md.info.get("longName") or ticker

# ── header ───────────────────────────────────────────────────────────────────
st.title(f"{company}  ·  {ticker}")
top = st.columns(4)
top[0].metric("Price", money(md.price, 2))
fv = detail.get("dcf_inputs", {})
base_fv = f.metrics.get("Base FV")
if base_fv and md.price:
    top[1].metric("DCF base fair value", money(base_fv, 0), f"{(base_fv/md.price-1)*100:+.0f}% vs price")
else:
    top[1].metric("DCF base fair value", money(base_fv, 0))
top[2].metric("Sector", md.sector or "—")
top[3].metric("Market cap", big(md.market_cap))

# verdict strip
vc = st.columns(4)
for col, res, name in zip(vc, [f, t, mk, (mc or _na_macro())], ["Fundamental", "Technical", "Markov", "Macro"]):
    col.markdown(f"**{name}**  \n{STANCE_EMOJI.get(res.stance,'⚪')} {res.verdict}")


tabs = st.tabs(["📋 Scorecard", "💰 Interactive DCF", "📑 Financials", "📈 Charts", "⚖️ Compare"])

# ── TAB 1: Scorecard ─────────────────────────────────────────────────────────
with tabs[0]:
    now = datetime.now(timezone.utc)
    run_obj = sc.assemble(ticker, f"{ticker} {now:%Y-%m-%d %H:%M}", now.isoformat(),
                          md.price, f, t, mk, mc or _na_macro())
    if run_obj.disagreement:
        st.warning(f"**Lenses disagree — the signal is here.**\n\n{run_obj.agreement_map.splitlines()[0]}")
    else:
        st.success(run_obj.agreement_map.splitlines()[0])

    for res, name in [(f, "1 · Fundamental"), (t, "2 · Technical"),
                      (mk, "3 · Markov regime"), (mc or _na_macro(), "4 · Macro / government")]:
        with st.expander(f"{STANCE_EMOJI.get(res.stance,'⚪')} {name} — {res.verdict}", expanded=True):
            st.write(res.summary)

    if config.AIRTABLE_API_KEY and st.button("💾 Save this run to Airtable"):
        try:
            from sinks import airtable_sink
            ids = airtable_sink.write_run(run_obj)
            st.success(f"Saved {sum(1 for v in ids.values() if v)}/{len(ids)} tables to Airtable.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Airtable save failed: {exc}")

# ── TAB 2: Interactive DCF ───────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Discounted cash-flow — drive the assumptions")
    di = detail.get("dcf_inputs", {})
    base_fcf0 = di.get("base_fcf")
    net_debt = di.get("net_debt") or 0
    shares = di.get("shares")
    g0 = di.get("rev_cagr")

    if not base_fcf0 or not shares:
        st.info("Not enough EDGAR data to build a DCF for this name "
                "(missing free cash flow or share count).")
    else:
        c = st.columns(5)
        base_fcf = c[0].number_input("Base annual FCF ($)", value=float(base_fcf0),
                                     step=float(abs(base_fcf0)) / 20 or 1.0, format="%.0f")
        growth = c[1].slider("FCF growth %/yr", -10.0, 30.0,
                             float(round((g0 or 0.05) * 100, 1)), 0.5) / 100
        discount = c[2].slider("Discount rate (WACC) %", 5.0, 15.0,
                               config.DCF_DISCOUNT_RATE * 100, 0.25) / 100
        term = c[3].slider("Terminal growth %", 0.0, 5.0,
                           config.DCF_TERMINAL_GROWTH * 100, 0.25) / 100
        years = c[4].slider("Projection years", 3, 10, config.DCF_YEARS)

        fv_live = fundamental.dcf_per_share(base_fcf, growth, discount, term, years, net_debt, shares)
        m = st.columns(3)
        if fv_live and md.price:
            m[0].metric("Fair value / share", money(fv_live, 2), f"{(fv_live/md.price-1)*100:+.0f}% vs price")
        else:
            m[0].metric("Fair value / share", money(fv_live, 2))
        m[1].metric("Current price", money(md.price, 2))
        m[2].metric("Implied EV", big((fv_live * shares + net_debt) if fv_live else None))

        st.caption("Sensitivity — fair value/share across growth (rows) × discount rate (cols)")
        g_axis = [growth + d for d in (-0.04, -0.02, 0, 0.02, 0.04)]
        d_axis = [max(term + 0.01, discount + d) for d in (-0.02, -0.01, 0, 0.01, 0.02)]
        grid = pd.DataFrame(
            [[fundamental.dcf_per_share(base_fcf, gg, dd, term, years, net_debt, shares) for dd in d_axis]
             for gg in g_axis],
            index=[f"{g*100:.1f}%" for g in g_axis],
            columns=[f"{d*100:.2f}%" for d in d_axis],
        )
        st.dataframe(grid.style.format(lambda v: money(v, 0)), width="stretch")
        st.caption("⚠️ A deliberately simple DCF on free annual data — explore the *relative* "
                   "sensitivity, don't read the absolute dollar value as a price target.")

# ── TAB 3: Financials ────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Annual financials & quality ratios (SEC EDGAR)")
    annual = detail.get("annual", [])
    if not annual:
        st.info(detail.get("error", "No financials parsed."))
    else:
        df = pd.DataFrame(annual).set_index("Year")
        disp = pd.DataFrame(index=df.index)
        disp["Revenue"] = df["Revenue"].map(big)
        disp["EBITDA"] = df["EBITDA"].map(big)
        disp["FCF"] = df["FCF"].map(big)
        disp["Gross M."] = df["Gross Margin"].map(pct)
        disp["Op M."] = df["Operating Margin"].map(pct)
        disp["Net M."] = df["Net Margin"].map(pct)
        disp["FCF M."] = df["FCF Margin"].map(pct)
        disp["ROIC"] = df["ROIC"].map(pct)
        disp["EPS"] = df["EPS"].map(lambda v: money(v, 2))
        disp["Net Debt"] = df["Net Debt"].map(big)
        st.dataframe(disp, width="stretch")

        st.caption("Current valuation vs the stock's own 5-year history (percentile; lower = cheaper)")
        vcols = st.columns(3)
        for col, label, key, pkey in zip(
                vcols, ["P/E", "EV/EBITDA", "EV/Sales"],
                ["PE", "EV EBITDA", "EV Sales"],
                ["PE 5y Pctile", "EV EBITDA 5y Pctile", "EV Sales 5y Pctile"]):
            col.metric(label, mult(f.metrics.get(key)), pct(f.metrics.get(pkey), 0) + " of 5y")

# ── TAB 4: Charts ────────────────────────────────────────────────────────────
with tabs[3]:
    if md.has_history:
        close = md.close
        st.subheader("Price & moving averages (≈6y)")
        chart = pd.DataFrame({
            "Close": close,
            "MA50": close.rolling(50).mean(),
            "MA200": close.rolling(200).mean(),
        })
        st.line_chart(chart)

        st.subheader("Drawdown from peak")
        dd = close / close.cummax() - 1.0
        st.area_chart(dd.rename("Drawdown"))

        hist = detail.get("history", {})
        if any(hist.values()):
            st.subheader("Valuation multiple — own history vs today")
            cols = st.columns(3)
            for col, key, label, cur in zip(
                    cols, ["pe", "ev_ebitda", "ev_sales"], ["P/E", "EV/EBITDA", "EV/Sales"],
                    [f.metrics.get("PE"), f.metrics.get("EV EBITDA"), f.metrics.get("EV Sales")]):
                pts = hist.get(key, [])
                if pts:
                    s = pd.Series({str(y): v for y, v in pts})
                    if cur is not None:
                        s["now"] = cur
                    col.caption(label)
                    col.bar_chart(s)
    else:
        st.info("No price history available for charts.")

# ── TAB 5: Compare ───────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Side-by-side comparison")
    peers = st.text_input("Tickers (comma-separated)", value=f"{ticker}, MSFT, NVDA")
    names = [x.strip().upper() for x in peers.replace(",", " ").split() if x.strip()]
    rows = []
    for tk in names:
        try:
            m2, f2, t2, mk2, d2 = load_core(tk)
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "Ticker": tk,
            "Price": money(m2.price, 2),
            "Fundamental": f"{STANCE_EMOJI.get(f2.stance,'⚪')} {f2.verdict}",
            "Technical": f"{STANCE_EMOJI.get(t2.stance,'⚪')} {t2.verdict}",
            "Markov": mk2.metrics.get("Current State", mk2.verdict),
            "P/E": mult(f2.metrics.get("PE")),
            "EV/EBITDA": mult(f2.metrics.get("EV EBITDA")),
            "ROIC": pct(f2.metrics.get("ROIC")),
            "Rev CAGR": pct(f2.metrics.get("Revenue CAGR 3y")),
            "12M ret": pct(t2.metrics.get("Return 12M")),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width="stretch")
