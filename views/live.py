"""Live Paper-Trading Lab — adaptive online ensemble, simulated, NO real orders.

Educational only — not financial advice. Fetches closed 1-minute candles every ~minute,
blends several online models by recent performance (Hedge weights), detects concept
drift, gates trades through real-trader FILTERS, manages risk (ATR stop, R:R target,
trailing, cooldown, kill-switch, vol-targeted sizing), and reports trader-grade metrics
vs THREE baselines (flat, buy-&-hold, naive). It is "tradeable" only if it beats ALL of
them — usually it won't, and it says so plainly. NO real orders are ever placed.
"""
import time

import pandas as pd
import streamlit as st

from lib import glossary, intraday, live, ui
from lib.intraday import WATCHLIST

dark = st.session_state.get("dark_mode", False)

st.markdown("## 🔴 Live Paper-Trading Lab")
st.caption("Adaptive online learning on 1-minute candles — **simulated paper trades only, "
           "never real orders.** Every number is measured live, net of costs.")

# --- permanent risk panel ---------------------------------------------------
with st.container(border=True):
    st.markdown("### 🛑 Risk reality check")
    rc = st.columns([1, 1, 2])
    lev = rc[0].slider("Leverage (×)", 1, 25, 4, key="live_lev")
    lm = intraday.leverage_math(lev)
    rc[1].metric("Liquidation distance", f"≈ {lm['liq_move_pct']:.1f}%")
    rc[2].warning(
        f"At **{lev}×**, ~**{lm['liq_move_pct']:.1f}%** against you can liquidate (less after "
        f"fees/maintenance). Funding accrues while holding. **~half of all trades will be wrong** "
        f"— survival comes from *small losses + position sizing*, not prediction. Even a real 55% "
        f"win-rate can lose money after costs.")

# --- controls ---------------------------------------------------------------
with st.container(border=True):
    st.markdown("##### ⚙️ Session settings")
    c = st.columns([2, 1, 1, 1])
    busy = st.session_state.get("live_active", False)
    sym = c[0].selectbox("Coin", list(WATCHLIST.keys()), index=0, disabled=busy)
    duration = c[1].selectbox("Duration (min)", [5, 15, 30, 60], index=2, disabled=busy)
    interval = c[2].selectbox("Tick (s)", [60, 90, 120, 180], index=1, disabled=busy)
    thr = c[3].slider("Signal threshold", 0.02, 0.20, 0.08, step=0.01,
                      help="Only paper-trade when blended P(up) deviates this far from 0.5.")
    capital = st.number_input("Account size (₹)", 1000, 100_000_000, 100_000, step=1000,
                              help="Used only to show the rupee risk per trade. No real money.")
    b1, b2 = st.columns(2)
    if not busy:
        if b1.button("▶️ Start live session", type="primary", width="stretch"):
            with st.spinner("Backfilling history + warming the online ensemble…"):
                live.ensure_session(sym)      # ccxt symbol (e.g. BTC) — NOT the coin id
                live.reset_session(sym)
            st.session_state.update(live_active=True, live_symbol=sym, live_thr=float(thr),
                                    live_cap=float(capital), live_end=time.time() + duration * 60,
                                    live_interval=int(interval))
            st.rerun()
    else:
        if b2.button("⏹️ Stop session", width="stretch"):
            st.session_state["live_active"] = False
            st.rerun()

active = st.session_state.get("live_active", False)
sym_id = st.session_state.get("live_symbol")
cap = float(st.session_state.get("live_cap", capital))
costs = dict(live.DEFAULT_COSTS)


def _pct(x, signed=False):
    return "—" if x is None else (f"{x*100:+.2f}%" if signed else f"{x*100:.1f}%")


def _num(x, fmt="{:.2f}"):
    return "—" if x is None else fmt.format(x)


def render_live():
    if not sym_id:
        st.info("Pick a coin and press **Start live session** to begin prequential paper-trading.")
        st.caption("📏 Educational only — not financial advice. Backtested/live edge does not guarantee future results.")
        return

    remaining = max(0, int(st.session_state.get("live_end", 0) - time.time()))
    if active and remaining <= 0:
        st.session_state["live_active"] = False

    res = live.tick(sym_id, float(st.session_state.get("live_thr", thr)), costs, cap)
    L, T = res.get("live", {}), res.get("trailing", {})

    head = st.columns([2, 1, 1])
    head[0].markdown(f"**{sym_id}/USDT** · {res.get('candles', 0)} candles · "
                     f"**{res.get('live_bars', 0)}** live bars")
    head[1].metric("Status", "🟢 LIVE" if active else "⏸️ stopped")
    head[2].metric("Time left", f"{remaining//60}m {remaining%60}s" if active else "—")

    # --- forward PAPER signal + regime + drift ---
    sig = L.get("next_signal", "NO-TRADE")
    sig_color = {"LONG": "#16a34a", "SHORT": "#ef4444"}.get(sig, "#9ca3af")
    with st.container(border=True):
        s1, s2 = st.columns([2, 1])
        with s1:
            st.markdown('<p class="bento-h">Next-bar PAPER signal (simulated)</p>', unsafe_allow_html=True)
            st.markdown(f'<span class="bento-big" style="color:{sig_color}">{sig}</span>', unsafe_allow_html=True)
            if L.get("next_p") is not None:
                st.caption(f"blended P(up) = {L['next_p']*100:.0f}% · threshold ±{int(float(st.session_state.get('live_thr', thr))*100)}%")
            if L.get("next_reason"):
                st.warning(L["next_reason"])
        with s2:
            reg = L.get("regime", "—")
            st.metric("Regime", reg, help=glossary.get_definition("Regime"))
            st.caption(f"Logic in use: **{L.get('logic', '—')}**")
            if L.get("drift"):
                st.error("⚠️ Drift detected — regime changed, ensemble re-balancing.")
            else:
                st.caption("Drift monitor: stable")

    # --- model weights (adaptive Hedge blend) ---
    w = L.get("weights") or T.get("weights")
    if w:
        st.markdown("##### 🧠 Online ensemble — current blend weights")
        st.caption("Each model is weighted by its **recent live loss** (Hedge / multiplicative weights); "
                   "stale models fade automatically.")
        st.bar_chart(pd.DataFrame({"weight": w}), height=160)

    # --- trade filters (why ticks were skipped) ---
    skips = L.get("skips") or {}
    if skips:
        st.markdown("##### 🚦 Trade filters — why most ticks are skipped")
        top = sorted(skips.items(), key=lambda kv: -kv[1])
        st.caption(" · ".join(f"**{k}** ×{v}" for k, v in top))
        st.caption("Real traders don't trade everything — dead-chop, fee-eating ranges, counter-trend "
                   "and low-volume setups are filtered out before any trade.")

    # --- live trader metrics ---
    st.markdown("##### 📊 Live session — trader-grade, net of costs")
    if L.get("n_trades", 0) < 1:
        st.info("No paper trades have cleared the filters yet (this is normal on 1-minute crypto — "
                "fees usually exceed the move). Metrics appear once trades clear. See the trailing "
                "backtest below for the warmed ensemble's behaviour.")
    else:
        m = st.columns(4)
        m[0].metric("Win-rate", _pct(L.get("win_rate")))
        m[1].metric("Expectancy/trade", _pct(L.get("expectancy"), signed=True), help=glossary.get_definition("Expectancy"))
        m[2].metric("Profit factor", _num(L.get("profit_factor")), help="Gross wins ÷ gross losses. >1 = profitable.")
        m[3].metric("Avg R", _num(L.get("avg_r"), "{:+.2f}"), help="Average reward-to-risk multiple per trade.")
        m2 = st.columns(4)
        m2[0].metric("Net Sharpe", _num(L.get("sharpe"), "{:.1f}"))
        m2[1].metric("Sortino", _num(L.get("sortino"), "{:.1f}"))
        m2[2].metric("Max DD", _pct(L.get("max_dd")))
        m2[3].metric("Paper trades", str(L.get("n_trades", 0)))
        m3 = st.columns(4)
        m3[0].metric("Avg MAE", _pct(L.get("avg_mae"), signed=True), help="Avg worst adverse excursion while in a trade.")
        m3[1].metric("Avg MFE", _pct(L.get("avg_mfe"), signed=True), help="Avg best favourable excursion while in a trade.")
        m3[2].metric("Win/Loss streak", f"{L.get('max_win_streak',0)} / {L.get('max_loss_streak',0)}")
        m3[3].metric("Avg hold", f"{L.get('avg_hold',0):.0f} bars")
        st.caption(f"Each paper trade risks ≈ **₹{cap*live.RISK_FRAC:,.0f}** "
                   f"({live.RISK_FRAC*100:.1f}% of account) — volatility-targeted sizing. Illustrative only.")

        # equity vs the 3 baselines
        if L.get("strat_eq") and L.get("eq_x"):
            df = pd.DataFrame({"paper account (risk-sized)": L["strat_eq"],
                               "buy & hold": L["bh_eq"], "flat": [1.0] * len(L["strat_eq"])},
                              index=L["eq_x"])
            st.line_chart(df, height=200)
        bl = st.columns(3)
        bl[0].metric("vs Flat", _pct(L.get("edge_per_bar"), signed=True), help="Strategy return per in-market bar.")
        bl[1].metric("Buy & hold (window)", _pct(L.get("baseline_buyhold"), signed=True))
        bl[2].metric("Naive (window)", _pct(L.get("baseline_naive"), signed=True))

        # honest verdict — must beat ALL baselines
        if L.get("killed"):
            st.error("🛑 Kill-switch hit — paper drawdown exceeded the limit. Trading halted; no edge.")
        elif L.get("beat_all"):
            st.success("So far this session **beat all three baselines** (flat, buy-&-hold, naive) — "
                       "still small-sample noise; keep risk tiny.")
        else:
            st.error("**No edge — No trade.** This session has NOT beaten all baselines "
                     "(flat / buy-&-hold / naive). That's the normal, honest outcome for 1-minute crypto.")

        # calibration drift warning
        if L.get("brier_recent") is not None and L["brier_recent"] > 0.25:
            st.warning(f"⚠️ Probabilities are poorly calibrated right now (recent {ui.term('Brier')} "
                       f"{L['brier_recent']:.3f} > 0.25). Don't trust the confidence until it recovers.")

    # --- calibration + trailing context ---
    with st.expander("🎯 Calibration (reliability) & trailing backtest"):
        rel = L.get("reliability") or T.get("reliability") or []
        if rel:
            st.caption(f"Brier {_num((L.get('brier') or T.get('brier')), '{:.3f}')} "
                       "(0 = perfect, 0.25 = no skill). Predicted vs actual by bin:")
            cal = pd.DataFrame({"predicted": [p for p, _, _ in rel], "actual": [a for _, a, _ in rel]}
                               ).set_index("predicted")
            st.line_chart(cal, height=160)
        if T.get("n_trades", 0) >= 1:
            st.caption(f"Trailing 12h paper backtest: win {_pct(T.get('win_rate'))} · "
                       f"expectancy {_pct(T.get('expectancy'), signed=True)} · PF {_num(T.get('profit_factor'))} · "
                       f"avg R {_num(T.get('avg_r'), '{:+.2f}')} · beat all baselines: {T.get('beat_all')}")
        else:
            st.caption("Trailing window has no completed trades yet (filters skipped fee-eating setups).")

    if active:
        st.caption(f"⏱️ auto-refreshing every {st.session_state.get('live_interval', interval)}s · "
                   "predict-then-update (prequential), no leakage · river ADWIN drift if installed.")
    st.caption("📏 Educational only — not financial advice. **NO real orders are ever placed.** "
               "Backtested/live edge does not guarantee future results.")


if active:
    @st.fragment(run_every=f"{st.session_state.get('live_interval', 90)}s")
    def _loop():
        render_live()
    _loop()
else:
    render_live()
