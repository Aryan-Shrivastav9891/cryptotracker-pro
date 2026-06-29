"""Intraday Signal Lab — short-horizon (1h/2h), cost-aware, honest edge scanner.

Runs the full ensemble + walk-forward backtest on HOURLY data and ranks coins by
their MEASURED net edge AFTER fees + slippage + funding. Nothing is hardcoded: a
coin is "tradeable" only if it beats a naive baseline out-of-sample after costs.

Accuracy only — not financial advice.
"""
import pandas as pd
import streamlit as st

from lib import glossary, intraday, ui
from lib.intraday import WATCHLIST

dark = st.session_state.get("dark_mode", False)

st.markdown("## ⚡ Intraday Signal Lab")
st.caption("Scan liquid coins on **1h / 2h** horizons with every installed algorithm, "
           "backtested out-of-sample **after trading costs**. It tells the truth — including "
           "when there's no edge.")

# --- PERMANENT leverage risk panel ------------------------------------------
with st.container(border=True):
    st.markdown("### 🛑 Leverage & risk")
    lc = st.columns([1, 2])
    leverage = lc[0].slider("Leverage (×)", 1, 25, 4)
    lm = intraday.leverage_math(leverage)
    lc[1].metric("Liquidation distance", f"≈ {lm['liq_move_pct']:.1f}% move against you")
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
         "Net Sharpe": "Skill score", "Coverage": "Confidence band",
         "Max DD": "Volatility"}


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
