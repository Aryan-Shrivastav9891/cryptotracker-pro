"""Live Paper-Trading Session — prequential online learning, simulated, NO real orders.

Educational only — not financial advice. Fetches closed 1-minute candles every
~minute, updates an online model predict-then-update (no leakage), paper-trades the
NEXT bar net of costs, and reports measured live hit-rate / win-rate / expectancy /
net Sharpe / calibration and SKILL vs a cost-aware naive baseline. It will usually
NOT beat naive — and it says so plainly.
"""
import time

import pandas as pd
import streamlit as st

from lib import glossary, intraday, live, ui
from lib.intraday import WATCHLIST

dark = st.session_state.get("dark_mode", False)

st.markdown("## 🔴 Live Paper-Trading Lab")
st.caption("Prequential online learning on 1-minute candles — **simulated paper trades only, "
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
    sym = c[0].selectbox("Coin", list(WATCHLIST.keys()), index=0,
                         disabled=st.session_state.get("live_active", False))
    duration = c[1].selectbox("Duration (min)", [5, 15, 30, 60], index=2,
                              disabled=st.session_state.get("live_active", False))
    interval = c[2].selectbox("Tick (s)", [60, 90, 120, 180], index=1,
                              disabled=st.session_state.get("live_active", False))
    thr = c[3].slider("Signal threshold", 0.02, 0.20, 0.08, step=0.01,
                      help="Only paper-trade when P(up) deviates this far from 0.5.")
    b1, b2 = st.columns(2)
    if not st.session_state.get("live_active", False):
        if b1.button("▶️ Start live session", type="primary", width="stretch"):
            with st.spinner("Backfilling history + warming the online model…"):
                live.ensure_session(WATCHLIST[sym])
                live.reset_session(WATCHLIST[sym])  # start the live window now
            st.session_state.update(live_active=True, live_symbol=WATCHLIST[sym],
                                    live_sym_label=sym, live_thr=float(thr),
                                    live_end=time.time() + duration * 60, live_interval=int(interval))
            st.rerun()
    else:
        if b2.button("⏹️ Stop session", width="stretch"):
            st.session_state["live_active"] = False
            st.rerun()

active = st.session_state.get("live_active", False)
sym_id = st.session_state.get("live_symbol")
sym_lbl = st.session_state.get("live_sym_label", "")
costs = dict(live.DEFAULT_COSTS)


def _pct(x, signed=False):
    return "—" if x is None else (f"{x*100:+.2f}%" if signed else f"{x*100:.1f}%")


def render_live():
    if not sym_id:
        st.info("Pick a coin and press **Start live session** to begin prequential paper-trading.")
        st.caption("📏 Educational only — not financial advice. Backtested/live edge does not guarantee future results.")
        return

    remaining = max(0, int(st.session_state.get("live_end", 0) - time.time()))
    if active and remaining <= 0:
        st.session_state["live_active"] = False

    res = live.tick(sym_id, float(st.session_state.get("live_thr", thr)), costs)
    livem, trail = res.get("live", {}), res.get("trailing", {})

    head = st.columns([2, 1, 1])
    head[0].markdown(f"**{sym_lbl}/USDT** · {res.get('candles', 0)} candles · "
                     f"**{res.get('live_bars', 0)}** live bars")
    head[1].metric("Status", "🟢 LIVE" if active else "⏸️ stopped")
    head[2].metric("Time left", f"{remaining//60}m {remaining%60}s" if active else "—")

    # --- forward PAPER signal (clearly simulated) ---
    sig = livem.get("next_signal", "NO-TRADE")
    sig_color = {"LONG": "#16a34a", "SHORT": "#ef4444"}.get(sig, "#9ca3af")
    with st.container(border=True):
        st.markdown('<p class="bento-h">Next-bar PAPER signal (simulated)</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="bento-big" style="color:{sig_color}">{sig}</span>', unsafe_allow_html=True)
        if livem.get("next_p") is not None:
            st.caption(f"model P(up) = {livem['next_p']*100:.0f}% · threshold ±{int(float(st.session_state.get('live_thr', thr))*100)}%")
        if livem.get("next_reason"):
            st.warning(livem["next_reason"])

    # --- live metrics (this session, measured net of costs) ---
    st.markdown("##### 📊 Live session — measured, net of costs")
    if livem.get("n_pred", 0) < 1:
        st.info("Collecting live bars… metrics appear after the first minute closes. "
                "Meanwhile, the trailing backtest below shows the warmed model's behaviour.")
    else:
        m = st.columns(4)
        m[0].metric("Hit-rate", _pct(livem.get("hit_rate")), help=glossary.get_definition("Directional accuracy"))
        m[1].metric("Win-rate", _pct(livem.get("win_rate")))
        m[2].metric("Expectancy/trade", _pct(livem.get("expectancy"), signed=True),
                    help=glossary.get_definition("Expectancy"))
        m[3].metric("Net Sharpe", "—" if livem.get("sharpe") is None else f"{livem['sharpe']:.1f}")
        m2 = st.columns(4)
        m2[0].metric("Skill vs naive", _pct(livem.get("skill_vs_naive"), signed=True),
                     help=glossary.get_definition("Skill score"))
        m2[1].metric("Brier", "—" if livem.get("brier") is None else f"{livem['brier']:.3f}",
                     help="Calibration error of the probabilities (lower is better; 0.25 = no skill).")
        m2[2].metric("Max DD", _pct(livem.get("max_dd")))
        m2[3].metric("Paper trades", str(livem.get("n_trades", 0)))
        if livem.get("equity_curve"):
            st.line_chart(pd.DataFrame({"paper equity (×)": livem["equity_curve"]}), height=180)
        # honest verdict
        if livem.get("killed"):
            st.error("🛑 Kill-switch hit — paper drawdown exceeded the limit. No edge; stop trading.")
        elif livem.get("beat_naive"):
            st.success("So far this session **beat the cost-aware naive baseline** — still noise; keep risk small.")
        else:
            st.error("**No edge — No trade.** This session has NOT beaten a cost-aware naive baseline. "
                     "That's the normal, honest outcome for 1-minute crypto.")

    # --- trailing context ---
    with st.expander("🔁 Trailing 12h paper backtest (model context)"):
        if trail.get("n_trades", 0) >= 1:
            st.caption(f"Over the warm-up window: hit {_pct(trail.get('hit_rate'))} · win {_pct(trail.get('win_rate'))} · "
                       f"expectancy {_pct(trail.get('expectancy'), signed=True)} · Brier "
                       f"{trail.get('brier') and round(trail['brier'],3)} · beat naive: {trail.get('beat_naive')}")
            rel = trail.get("reliability", [])
            if rel:
                st.caption("Calibration (predicted vs actual): " +
                           " · ".join(f"{int(p*100)}%→{int(a*100)}% (n={k})" for p, a, k in rel))
        else:
            st.caption("Not enough history yet.")

    if active:
        st.caption(f"⏱️ auto-refreshing every {st.session_state.get('live_interval', interval)}s · "
                   "predict-then-update (prequential), no leakage.")
    st.caption("📏 Educational only — not financial advice. NO real orders are ever placed. "
               "Backtested/live edge does not guarantee future results.")


if active:
    @st.fragment(run_every=f"{st.session_state.get('live_interval', 90)}s")
    def _loop():
        render_live()
    _loop()
else:
    render_live()
