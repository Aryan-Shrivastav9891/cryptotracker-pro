"""Intraday Signal Lab — short-horizon (1h/2h), cost-aware, honest edge scanner.

Runs the full ensemble + walk-forward backtest on HOURLY data and ranks coins by
their MEASURED net edge AFTER fees + slippage + funding. Nothing is hardcoded: a
coin is "tradeable" only if it beats a naive baseline out-of-sample after costs.

Accuracy only — not financial advice.
"""
import pandas as pd
import streamlit as st

from lib import glossary, intraday, playbook, ui
from lib.intraday import WATCHLIST

dark = st.session_state.get("dark_mode", False)

st.markdown("## ⚡ Intraday Signal Lab")
st.caption("Scan liquid coins on **1h / 2h** horizons with every installed algorithm, "
           "backtested out-of-sample **after trading costs**. It tells the truth — including "
           "when there's no edge.")

# --- PERMANENT leverage risk panel ------------------------------------------
with st.container(border=True):
    st.markdown("### 🛑 Leverage & risk")
    lc = st.columns([1, 1, 1])
    leverage = lc[0].slider("Leverage (×)", 1, 25, 4)
    capital = lc[1].number_input("Account size (₹)", 1000, 100_000_000, 100_000, step=1000)
    lm = intraday.leverage_math(leverage)
    lc[2].metric("Liquidation distance", f"≈ {lm['liq_move_pct']:.1f}%")
    st.warning(
        f"At **{leverage}×**, a move of only **~{lm['liq_move_pct']:.1f}%** against your position "
        f"can liquidate the whole thing (even less after fees + maintenance margin). "
        f"**Funding fees accrue the entire time you hold.** Intraday crypto is dominated by noise — "
        f"even a *real* 55% directional edge can still **lose money after costs**. Size positions accordingly."
    )

# --- Controls ---------------------------------------------------------------
with st.container(border=True):
    st.markdown("##### ⚙️ Scan settings")
    c = st.columns([3, 1, 1])
    syms = c[0].multiselect("Watchlist", list(WATCHLIST.keys()),
                            default=["BTC", "ETH", "SOL", "BNB", "XRP"])
    hours = c[1].selectbox("History (hours)", [360, 720, 1440], index=1,
                           format_func=lambda h: f"{h}h (~{h//24}d)")
    fast = c[2].toggle("Fast mode", value=True,
                       help="Skip the slow tree/SVR models for a quicker scan.")
    with st.expander("💸 Cost assumptions (per trade)"):
        cc = st.columns(3)
        taker = cc[0].number_input("Taker fee %", 0.0, 0.5, 0.05, step=0.01) / 100.0
        slip = cc[1].number_input("Slippage %", 0.0, 0.5, 0.05, step=0.01) / 100.0
        funding = cc[2].number_input("Funding % / 8h", 0.0, 0.2, 0.01, step=0.005,
                                     format="%.3f") / 100.0
    costs_tuple = (taker, slip, funding)

if st.button("🚀 Run intraday scan", type="primary"):
    st.session_state["intraday_run"] = True

if not st.session_state.get("intraday_run"):
    st.info("Pick a watchlist and press **Run intraday scan**. The first scan trains many "
            "models per coin on hourly data (slow, ~30–60s each) and is then cached.")
    st.caption("📏 Accuracy only — not financial advice.")
    st.stop()

if not syms:
    st.warning("Select at least one coin.")
    st.stop()

# --- Run the scan (cached per coin) -----------------------------------------
scanned = []
prog = st.progress(0.0, text="Starting…")
for i, sym in enumerate(syms):
    prog.progress(i / len(syms), text=f"Backtesting {sym}… ({i + 1}/{len(syms)})")
    r = intraday.scan_coin(sym, WATCHLIST[sym], int(hours), costs_tuple, bool(fast))
    if r:
        scanned.append(r)
prog.progress(1.0, text="Done")
prog.empty()

if not scanned:
    st.warning("Could not fetch enough hourly history for the selected coins. Try again shortly.")
    st.stop()

SIG_EMOJI = {"Long": "🟢 Long", "Short": "🔴 Short", "No-edge": "⚪ No-edge"}


def _pct(x, signed=False):
    return ("—" if x is None else (f"{x*100:+.2f}%" if signed else f"{x*100:.1f}%"))


def build_table(label):
    rows = []
    for r in scanned:
        hd = r["results"].get(label)
        if not hd:
            rows.append({"Coin": r["symbol"], "Signal": "—", "_sort": -9})
            continue
        rows.append({
            "Coin": r["symbol"],
            "Signal": SIG_EMOJI.get(hd["signal"], hd["signal"]),
            "Directional": _pct(hd["directional"]),
            "Skill": _pct(hd["skill"], signed=True),
            "Net move (cost-adj)": _pct(hd["expected_net"], signed=True),
            "Net/trade": _pct(hd["net_mean"], signed=True),
            "Net Sharpe": "—" if hd["sharpe"] is None else f"{hd['sharpe']:.2f}",
            "Defl. Sharpe": "—" if hd.get("deflated_sharpe") is None else f"{hd['deflated_sharpe']:.2f}",
            "Max DD": _pct(hd["maxdd"]),
            "Coverage": _pct(hd["coverage"]),
            "Tradeable?": "✅ yes" if hd["tradeable"] else "no",
            "_sort": hd["net_mean"] if hd["net_mean"] is not None else -9,
        })
    rows.sort(key=lambda x: x["_sort"], reverse=True)
    for x in rows:
        x.pop("_sort", None)
    return pd.DataFrame(rows)


_COLS = {"Directional": "Directional accuracy", "Skill": "Skill score",
         "Net Sharpe": "Skill score", "Defl. Sharpe": "Deflated Sharpe",
         "Coverage": "Confidence band", "Max DD": "Volatility"}


def _fmt_px(x):
    if x is None:
        return "—"
    ax = abs(x)
    return f"{x:,.0f}" if ax >= 100 else (f"{x:,.2f}" if ax >= 1 else f"{x:,.6f}")


def render_playbook_card(symbol, pb, capital):
    with st.container(border=True):
        if not pb or not pb.get("available"):
            st.markdown(f"**{symbol}** — _playbook unavailable_")
            return
        regime = pb["regime"]
        reg_color = "#22c55e" if regime == "Trend" else "#3b82f6"
        st.markdown(
            f"**{symbol}** &nbsp; <span class='ctp-badge' style='background:{reg_color}22;"
            f"color:{reg_color}'>{ui.term('Regime')}: {regime}</span>", unsafe_allow_html=True)
        style = pb.get("style")
        if style in (None, "No-trade"):
            st.markdown("<span class='ctp-badge' style='background:#9ca3af22;color:#9ca3af'>"
                        "⚪ No-trade</span>", unsafe_allow_html=True)
            st.caption(pb.get("reason", ""))
            return
        if pb.get("skip_low_conf"):
            st.markdown("<span class='ctp-badge' style='background:#eab30822;color:#ca8a04'>"
                        "⏸ Skip — below confidence threshold</span>", unsafe_allow_html=True)
        sidec = "#16a34a" if pb["side"] == "Long" else "#ef4444"
        styled = ui.term(pb["style"]) if pb["style"] in ("Breakout",) else pb["style"]
        st.markdown(f"<span class='ctp-badge' style='background:{sidec}22;color:{sidec}'>"
                    f"{styled} · {pb['side']}</span>", unsafe_allow_html=True)
        st.caption(f"Suggested: {pb['entry_text']}.")
        st.markdown(
            f"- Entry ≈ **${_fmt_px(pb['entry_zone'])}** · "
            f"{ui.term('Stop-loss')} **${_fmt_px(pb['stop'])}** ({pb['sl_pct']*100:.1f}%) · "
            f"{ui.term('Take-profit')} **${_fmt_px(pb['target'])}** ({pb['tp_pct']*100:.1f}%)",
            unsafe_allow_html=True)
        prob, wr, exp = pb.get("probability"), pb.get("win_rate"), pb.get("expectancy")
        st.markdown(
            f"- {ui.term('Meta-label probability')} "
            f"**{round(prob*100) if prob is not None else '—'}%** · win-rate "
            f"**{round(wr*100) if wr is not None else '—'}%** · "
            f"{ui.term('Expectancy')} **{exp*100:+.2f}%/trade** (net, OOS, n={pb.get('n_test')})",
            unsafe_allow_html=True)
        sz = playbook.position_sizing(pb["sl_pct"], wr, prob, capital)
        st.caption(
            f"Illustrative size: risk if {ui.term('Stop-loss')} hits ≈ ₹{sz['rupee_risk']:,.0f} "
            f"({sz['risk_fraction_pct']:.1f}% of account) · cap ≤ {sz['frac_kelly_pct']:.1f}% of "
            f"account ({ui.term('Kelly')} ¼-cap). Not advice.", unsafe_allow_html=True)


def show_horizon(label):
    df = build_table(label)
    tradeable_n = sum(1 for r in scanned if (r["results"].get(label) or {}).get("tradeable"))
    if tradeable_n == 0:
        st.error(f"**No tradeable edge found right now** on the {label} horizon — every coin's "
                 f"cost-adjusted, out-of-sample result fails to beat a naive baseline. "
                 f"Sitting out is the honest call.")
    else:
        st.success(f"{tradeable_n} coin(s) show a measured, cost-adjusted edge on {label} "
                   f"(still risky — see the leverage panel).")
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={col: st.column_config.TextColumn(col, help=glossary.get_definition(term))
                       for col, term in _COLS.items()},
    )

    # --- Entry Playbook (regime-driven dip vs breakout; only shown when it has edge) ---
    st.markdown(f"##### 🎯 Entry Playbook · {label}")
    st.markdown(ui.annotate(
        "Dip-buy = buying after price pulls back toward support, expecting a bounce — works in "
        "Range markets, dangerous in downtrends. Breakout = buying as price pushes past a recent "
        "high, expecting continuation — works in Trend markets, prone to false breakouts in chop. "
        "The card surfaces a style ONLY if it beat a cost-aware naive baseline out-of-sample for "
        "that coin; otherwise No-trade."), unsafe_allow_html=True)
    st.caption(f"⚠️ Illustrative levels from a backtest — NOT a recommendation to trade. Backtested "
               f"edge does not guarantee future results; intraday crypto is dominated by noise. "
               f"At {leverage}×, a ~{lm['liq_move_pct']:.0f}% adverse move (less after fees) can liquidate.")
    pcols = st.columns(2)
    for i, r in enumerate(scanned):
        with pcols[i % 2]:
            render_playbook_card(r["symbol"], r.get("playbooks", {}).get(label), capital)


t1, t2 = st.tabs(["⏱️ 1h horizon", "⏱️ 2h horizon"])
with t1:
    show_horizon("1h")
with t2:
    show_horizon("2h")

with st.expander("📖 How to read this (and why most rows say No-edge)"):
    st.markdown(
        "- **Directional** = how often the model called the next move's up/down (50% = coin toss).\n"
        "- **Skill** = % lower error than a naive baseline, out-of-sample (>0 = real edge).\n"
        "- **Net move / Net per trade** = expected/realised return **after** fees + slippage + funding.\n"
        "- **Net Sharpe / Max DD** = risk-adjusted return and worst peak-to-trough of the cost-aware backtest.\n"
        "- **Coverage** = share of real prices that landed inside the 80% conformal band (target 80%).\n"
        "- A coin is **Tradeable** only if it beats the cost-aware naive baseline out-of-sample. "
        "Intraday crypto is mostly noise, so **No-edge is the normal, honest result**.")

src = scanned[0]["source"] if scanned else "?"
st.caption(f"Data: {src} · {len(scanned[0]['models_available'])} models per coin · "
           "results are real out-of-sample walk-forward backtests, costs included.")
st.caption("📏 Accuracy only — not financial advice.")
