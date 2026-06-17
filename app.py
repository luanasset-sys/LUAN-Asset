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
from data import fred, llm
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


@st.cache_data(show_spinner="Running macro web search…", ttl=900)
def load_macro(ticker: str):
    md = fetch_market_data(ticker)
    return macro.run(md)


@st.cache_data(ttl=1800, show_spinner=False)
def rates_snap():
    return fred.rates_snapshot()


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


tabs = st.tabs(["📋 Scorecard", "💰 Valuation & DCF", "📝 Research Report", "🌐 Macro & News",
                "📑 Financials", "📈 Charts", "⚖️ Compare"])

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

# ── TAB 1: Valuation & DCF ───────────────────────────────────────────────────
with tabs[1]:
    di = detail.get("dcf_inputs", {})
    base_fcf0 = di.get("base_fcf")
    net_debt = di.get("net_debt") or 0
    shares = di.get("shares")
    g0 = di.get("rev_cagr")

    if not base_fcf0 or not shares:
        st.info("Not enough EDGAR data to build a DCF for this name "
                "(missing free cash flow or share count).")
    else:
        st.subheader("Drive the assumptions")
        wacc_info = detail.get("wacc", {})
        wacc_default = (wacc_info.get("wacc") or config.DCF_DISCOUNT_RATE) * 100
        with st.expander(f"🏗️ How the discount rate (bottom-up WACC ≈ {wacc_default:.1f}%) is built"):
            if wacc_info:
                st.markdown(
                    f"- **Cost of equity** (CAPM): risk-free {pct(wacc_info['rf'])} + beta "
                    f"{wacc_info['beta']:.2f} × equity-risk-premium {pct(wacc_info['erp'])} "
                    f"= **{pct(wacc_info['ke'])}**\n"
                    f"- **Cost of debt** (after {pct(wacc_info['tax'])} tax): "
                    f"**{pct(wacc_info['kd_after_tax'])}**\n"
                    f"- **Weights**: equity {pct(wacc_info['weight_equity'])}, "
                    f"debt {pct(wacc_info['weight_debt'])}\n"
                    f"- **→ WACC ≈ {pct(wacc_info['wacc'])}** (bounded to 6–14%)")
            st.caption("Bottom-up estimate from free data (live treasury yield + the stock's beta). "
                       "Adjust the slider if you have a better number.")

        c = st.columns(5)
        base_fcf = c[0].number_input("Base annual FCF ($)", value=float(base_fcf0),
                                     step=float(abs(base_fcf0)) / 20 or 1.0, format="%.0f")
        growth = c[1].slider("Stage-1 FCF growth %/yr", -10.0, 40.0,
                             float(round((g0 or 0.06) * 100, 1)), 0.5) / 100
        discount = c[2].slider("Discount rate (WACC) %", 5.0, 15.0,
                               float(round(wacc_default, 2)), 0.25) / 100
        term = c[3].slider("Terminal growth %", 0.0, 5.0,
                           config.DCF_TERMINAL_GROWTH * 100, 0.25) / 100
        years = c[4].slider("Projection years", 5, 15, 10)

        FADE = True  # stage-1 growth fades linearly to terminal over the horizon
        fv_live = fundamental.dcf_per_share(base_fcf, growth, discount, term, years, net_debt, shares, FADE)
        impl = fundamental.implied_growth(md.price, base_fcf, discount, term, years, net_debt, shares, FADE)

        m = st.columns(3)
        if fv_live and md.price:
            m[0].metric("Your DCF fair value", money(fv_live, 2),
                        f"{(fv_live/md.price-1)*100:+.0f}% vs price")
        else:
            m[0].metric("Your DCF fair value", money(fv_live, 2))
        m[1].metric("Current price", money(md.price, 2))
        m[2].metric("Market-implied growth", pct(impl, 1) if impl is not None else "off the charts",
                    help="Reverse DCF: the FCF growth today's price implies, holding your other "
                         "assumptions fixed.")

        # The headline insight — what the market is pricing in
        if impl is not None:
            gap = ""
            if g0 is not None:
                gap = (f" — vs the company's ~{pct(g0,1)} recent trailing growth. "
                       + ("The market is betting it **accelerates**." if impl > g0 + 0.005
                          else "The market expects it to **slow**." if impl < g0 - 0.005
                          else "Roughly **in line** with its history."))
            st.info(f"💡 **What the market is pricing in:** at **{money(md.price,2)}**, today's price "
                    f"implies a starting **~{pct(impl,1)} FCF growth, fading to {pct(term,1)} over "
                    f"{years} years**, discounted at your {pct(discount,1)} WACC.{gap}")
        else:
            st.warning("⚠️ Today's price can't be reached by this DCF even at extreme growth — its value "
                       "rests on things this simple model doesn't capture (buybacks, a much longer growth "
                       "runway, optionality). Lean on the multiples and the *direction* of these numbers, "
                       "not the absolute target.")

        # Where the value comes from — transparent breakdown
        bd = fundamental.dcf_breakdown(base_fcf, growth, discount, term, years, net_debt, shares, FADE)
        if bd:
            st.subheader("Where the value comes from")
            bc = st.columns(2)
            proj = pd.DataFrame(bd["rows"]).set_index("Year")
            bc[0].caption("Each year's growth, projected free cash flow, and its worth today")
            bc[0].dataframe(pd.DataFrame({
                "Growth": proj["Growth"].map(lambda v: pct(v)),
                "Projected FCF": proj["Projected FCF"].map(big),
                "Present value": proj["Present value"].map(big),
            }), width="stretch")
            bridge = pd.DataFrame({
                "Step": ["PV of explicit FCF", "+ PV of terminal value", "= Enterprise value",
                         "− Net debt", "= Equity value", "÷ shares → per share"],
                "Value": [big(bd["pv_explicit"]), big(bd["pv_terminal"]), big(bd["enterprise_value"]),
                          big(bd["net_debt"]), big(bd["equity_value"]), money(bd["per_share"], 2)],
            }).set_index("Step")
            bc[1].caption("The bridge from cash flows to a per-share value")
            bc[1].dataframe(bridge, width="stretch")
            if bd["terminal_pct_of_value"] is not None:
                st.caption(f"⚠️ {pct(bd['terminal_pct_of_value'],0)} of the value sits in the terminal "
                           f"value — everything beyond year {years}. The more value is out there, the "
                           f"more the answer depends on guesses about the distant future.")

        # Backlog / remaining performance obligations
        bl = detail.get("backlog", {})
        if bl.get("latest"):
            st.subheader("Backlog (contracted future revenue)")
            blc = st.columns(2)
            blc[0].metric("Reported backlog (RPO)", big(bl["latest"]),
                          (pct(bl["coverage"], 0) + " of annual revenue") if bl.get("coverage") else None)
            pts = bl.get("series", [])
            if len(pts) >= 2:
                blc[1].caption("Backlog by year")
                blc[1].bar_chart(pd.Series({str(y): v for y, v in pts}))
            st.caption("Backlog = signed business not yet booked as revenue — a forward demand signal. "
                       "Rising backlog and high coverage support a higher growth assumption above. "
                       "(Only shown for companies that report remaining performance obligations.)")

        # Sensitivity
        st.subheader("Sensitivity")
        st.caption("Fair value / share across growth (rows) × discount rate (cols)")
        g_axis = sorted({round(growth + d, 4) for d in (-0.04, -0.02, 0, 0.02, 0.04)})
        d_axis = sorted({round(max(term + 0.0101, discount + d), 4)
                         for d in (-0.02, -0.01, 0, 0.01, 0.02)})
        grid = pd.DataFrame(
            [[fundamental.dcf_per_share(base_fcf, gg, dd, term, years, net_debt, shares, FADE)
              for dd in d_axis] for gg in g_axis],
            index=[f"{g*100:.1f}%" for g in g_axis],
            columns=[f"{d*100:.2f}%" for d in d_axis],
        )
        st.dataframe(grid.style.format(lambda v: money(v, 0)), width="stretch")

        # AI explanation of the valuation (optional — needs Anthropic key)
        st.subheader("Why is it valued like this?")
        if llm.available():
            if st.button("🧠 Explain this valuation in plain English"):
                with st.spinner("Thinking…"):
                    txt = llm.synthesize(
                        f"In 4-6 plain-English sentences, explain why {company} ({ticker}) trades where it "
                        f"does and what investors are betting on. Use ONLY these facts; invent nothing. "
                        f"Price {money(md.price,2)}; simple-DCF value {money(fv_live,2)}; market-implied FCF "
                        f"growth {pct(impl,1) if impl is not None else 'extreme/off-model'} vs ~{pct(g0,1)} "
                        f"trailing; P/E {mult(f.metrics.get('PE'))} "
                        f"({pct(f.metrics.get('PE 5y Pctile'),0)} of its own 5-yr range); "
                        f"EV/EBITDA {mult(f.metrics.get('EV EBITDA'))}; ROIC {pct(f.metrics.get('ROIC'))}; "
                        f"FCF margin {pct(f.metrics.get('FCF Margin'))}; "
                        f"backlog {big(bl.get('latest')) if bl.get('latest') else 'n/a'}. "
                        f"Explain what would have to be true to justify the price. Do not give buy/sell advice.",
                        max_tokens=400)
                    st.write(txt or "Explanation unavailable right now.")
        else:
            st.caption("Add your Anthropic key (`ANTHROPIC_API_KEY` in `.env`) to get an AI-written "
                       "explanation of the valuation here.")

        st.caption("⚠️ A deliberately simple DCF on free annual data — best used for the *reverse* read "
                   "(what's priced in) and *relative* sensitivity, not as a literal price target.")

# ── TAB 2: Research Report (Claude-written) ──────────────────────────────────
with tabs[2]:
    st.subheader(f"Equity research report — {company} ({ticker})")
    if not llm.available():
        st.info("The full written report runs on the Anthropic API. Add `ANTHROPIC_API_KEY` to your "
                "`.env` (with a little account credit), save, and restart — then generate it here.")
    else:
        st.caption("Claude writes a full report from the computed model + live web search: business & "
                   "moat, valuation & fair-value range, bull/bear & risks, macro/policy/catalysts. "
                   "~30–60s; uses your Anthropic key.")
        key = f"report::{ticker}"
        if st.button("📝 Generate / regenerate report"):
            di = detail.get("dcf_inputs", {})
            wv = (detail.get("wacc", {}) or {}).get("wacc") or 0.085
            impl_ctx = fundamental.implied_growth(md.price, di.get("base_fcf"), wv, 0.03, 10,
                                                  di.get("net_debt") or 0, di.get("shares"), True)
            ctx = (
                f"Company {company} ({ticker}); sector {md.sector}. Price {money(md.price,2)}; "
                f"market cap {big(md.market_cap)}. Bottom-up WACC {pct(wv)}. "
                f"Revenue {big(f.metrics.get('Revenue TTM'))}; revenue CAGR "
                f"{pct(f.metrics.get('Revenue CAGR 3y'))}; EPS CAGR {pct(f.metrics.get('EPS CAGR 3y'))}. "
                f"Gross/op/net margin {pct(f.metrics.get('Gross Margin'))}/"
                f"{pct(f.metrics.get('Operating Margin'))}/{pct(f.metrics.get('Net Margin'))}; "
                f"FCF margin {pct(f.metrics.get('FCF Margin'))}; ROIC {pct(f.metrics.get('ROIC'))}; "
                f"net debt/EBITDA {mult(f.metrics.get('Net Debt to EBITDA'))}. "
                f"P/E {mult(f.metrics.get('PE'))} ({pct(f.metrics.get('PE 5y Pctile'),0)} of own 5y), "
                f"EV/EBITDA {mult(f.metrics.get('EV EBITDA'))}, EV/Sales {mult(f.metrics.get('EV Sales'))}. "
                f"Backlog {big(detail.get('backlog',{}).get('latest'))}. "
                f"Reverse-DCF: the price implies ~"
                f"{pct(impl_ctx,1) if impl_ctx is not None else 'extreme/off-model'} stage-1 FCF growth. "
                f"Technical read: {t.verdict}. Markov regime: {mk.verdict}."
            )
            prompt = (
                f"Write a thorough equity research report on {company} ({ticker}) for a sophisticated "
                f"investor. Use web search for current business facts, recent news, competitive "
                f"position, and macro/policy/geopolitical context; cite sources with dates and URLs. "
                f"Ground all figures in this computed model data and do not contradict it:\n\n{ctx}\n\n"
                f"Use these markdown sections:\n"
                f"## Business & moat\n"
                f"## Valuation & fair value — reason explicitly from the DCF, the bottom-up WACC "
                f"({pct(wv)}), the multiples-vs-own-history, and the reverse-DCF; conclude with a "
                f"defensible fair-value RANGE and exactly what must be true to justify it\n"
                f"## Bull case\n## Bear case\n## Key risks\n## Macro, policy & catalysts\n"
                f"## Bottom line — a reasoned synthesis, NOT buy/sell advice\n\n"
                f"Be specific and concrete; avoid generic filler. End with exactly: "
                f"'Decision-support only — not financial advice.'"
            )
            with st.spinner("Researching and writing… (~30–60s)"):
                st.session_state[key] = llm.web_research(prompt, max_tokens=7000) or "Report unavailable."
        if st.session_state.get(key):
            st.markdown(st.session_state[key])

# ── TAB 3: Macro & News ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Macro, policy, geopolitics & live news — {ticker}")
    r = rates_snap()
    rc = st.columns(4)
    rc[0].metric("Fed funds", f"{r['fed_funds']:.2f}%" if r.get("fed_funds") is not None else "—")
    rc[1].metric("US 10Y", f"{r['ust_10y']:.2f}%" if r.get("ust_10y") is not None else "—")
    rc[2].metric("US 2Y", f"{r['ust_2y']:.2f}%" if r.get("ust_2y") is not None else "—")
    rc[3].metric("CPI YoY", f"{r['cpi_yoy']:.1f}%" if r.get("cpi_yoy") is not None else "—")
    st.divider()

    if not llm.available():
        st.info("Live policy / geopolitics / news uses web search via the Anthropic API. Add "
                "`ANTHROPIC_API_KEY` to your `.env` to switch it on. (FRED rates above always work.)")
    else:
        if st.button("🌐 Fetch / refresh live news & policy"):
            load_macro.clear()
            st.session_state["macro_on"] = True
        if mc is not None or st.session_state.get("macro_on"):
            res = load_macro(ticker)
            parsed = res.raw.get("parsed", {}) if res and res.raw else {}
            badge = {"Net Tailwind": "🟢", "Net Headwind": "🔴"}.get(res.verdict, "⚪")
            st.markdown(f"### {badge} Overall: {res.verdict}")
            if parsed.get("rates_posture"):
                st.markdown(f"**Rates posture:** {parsed['rates_posture']}")
            if parsed.get("geopolitics"):
                st.markdown(f"**Geopolitics / conflict:** {parsed['geopolitics']}")
            if parsed.get("forward_view"):
                st.markdown(f"**Forward outlook:** {parsed['forward_view']}")
            sigs = parsed.get("signals", [])
            if sigs:
                st.markdown("#### Signals & headlines")
                for sg in sigs:
                    imp = str(sg.get("impact", "neutral")).lower()
                    b = {"tailwind": "🟢 Tailwind", "headwind": "🔴 Headwind"}.get(imp, "⚪ Neutral")
                    hz = sg.get("horizon", "")
                    hz = f" · {hz}-term" if hz else ""
                    head = sg.get("headline") or sg.get("item", "")
                    src = sg.get("source")
                    head_md = f"[{head}]({src})" if src else head
                    line = f"**{b}{hz}** — {head_md}  \n_{sg.get('date','')}_ — {sg.get('rationale','')}"
                    if src:
                        line += f"  \n[🔗 Read article →]({src})"
                    st.markdown(line)
            st.caption("Pulled live from the web at fetch time. Click the button to refresh for the "
                       "latest. (~20–40s per fetch; uses your Anthropic key.)")
        else:
            st.caption(f"Click **Fetch / refresh** to pull live policy, geopolitics, and headlines for "
                       f"{ticker} and how they bear on its forward outlook.")

# ── TAB 4: Financials ────────────────────────────────────────────────────────
with tabs[4]:
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

# ── TAB 5: Charts ────────────────────────────────────────────────────────────
with tabs[5]:
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

# ── TAB 6: Compare ───────────────────────────────────────────────────────────
with tabs[6]:
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
