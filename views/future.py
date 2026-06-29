"""Future Gains — AI Forecast Lab (multi-horizon, backtested, honest).

Shows forecasts at 1D / 4D / 1W / 1M, each validated out-of-sample with its own
MAPE / directional / skill-vs-naive, and flagged unreliable when it has no edge.
Bentobox layout: combined chart, per-horizon table, news + sentiment, plain-English note.
"""
import pandas as pd
import streamlit as st

from lib import charts, coingecko, forecast, news, ui
from lib.formatting import format_inr

dark = st.session_state.get("dark_mode", False)

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); height: 100%;
    }
    .bento-h { font-size:0.8rem; text-transform:uppercase; letter-spacing:.04em; opacity:.7; margin:0; }
    .bento-big { font-size:1.9rem; font-weight:800; line-height:1.1; margin:.1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

SIG_COLORS = {"Strong Buy": "#16a34a", "Buy": "#22c55e", "Hold": "#3b82f6",
              "Sell": "#f97316", "Strong Sell": "#ef4444"}

st.markdown("## ⏳ AI Forecast Lab")
st.caption("Forecasts at **1D · 4D · 1W · 1M**, each backtested on unseen data. "
           "Beats a naive baseline — or it tells you it doesn't.")

# --- controls ---------------------------------------------------------------
market = coingecko.load_markets(per_page=100, sparkline=False)
if not market:
    st.warning("No market data available right now.")
    st.stop()

by_id = {c.get("id"): c for c in market}
labels = {f"{c.get('name')} ({(c.get('symbol') or '').upper()})": c.get("id") for c in market}
label_list = list(labels.keys())

preselected = st.session_state.get("selected_coin_id")
default_index = 0
if preselected in by_id:
    for i, lbl in enumerate(label_list):
        if labels[lbl] == preselected:
            default_index = i
            break

with st.container(border=True):
    ctrl = st.columns([3, 1])
    chosen_label = ctrl[0].selectbox("Coin", label_list, index=default_index)
    coin_id = labels[chosen_label]
    days = ctrl[1].selectbox("History", [365, 540, 730], index=1, format_func=lambda d: f"{d} days")

coin = by_id.get(coin_id, {})
name = coin.get("name", coin_id)
symbol = (coin.get("symbol") or "").upper()

# --- train + backtest (all horizons) + sentiment ----------------------------
with st.status(f"Training ensemble for {name} across all horizons…", expanded=True) as status:
    st.write(f"📥 Loading ~{days} days of daily prices (Binance → CoinGecko fallback)…")
    st.write("🧪 Walk-forward backtest, h-step-ahead, for 1D / 4D / 1W / 1M…")
    result = forecast.forecast_coin(coin_id, symbol, days=days)
    if not result:
        status.update(label="Not enough history to backtest", state="error")
        st.warning(f"Not enough daily history to backtest {name} honestly (need ~70+ daily "
                   f"candles). Try a longer history window, or another coin.")
        st.stop()
    st.write("📰 Fetching news + scoring sentiment…")
    sent = news.news_sentiment(symbol, name=name, limit=8)
    status.update(label=f"Done · {result['engine']} · data: {result['source']}", state="complete")

horizons = result["horizons"]
current = result["current_price"]

# --- BENTO ROW 1: combined chart (wide) + sentiment (narrow) ----------------
left, right = st.columns([2, 1])
with left:
    with st.container(border=True):
        st.markdown('<p class="bento-h">🔮 Forecast — 1D / 4D / 1W / 1M (with 80% bands)</p>',
                    unsafe_allow_html=True)
        fig = charts.multi_forecast_chart(result["history_dates"], result["history_prices"],
                                          horizons, dark=dark, height=420)
        st.plotly_chart(fig, use_container_width=True, key=f"mfc_{coin_id}_{days}")
with right:
    with st.container(border=True):
        st.markdown('<p class="bento-h">Current price</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="bento-big">{format_inr(current)}</span>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<p class="bento-h">📰 News sentiment</p>', unsafe_allow_html=True)
        if sent["available"]:
            s_color = {"Bullish": "#16a34a", "Bearish": "#ef4444"}.get(sent["label"], "#3b82f6")
            st.markdown(f'<span class="bento-big" style="color:{s_color}">{sent["label"]}</span>',
                        unsafe_allow_html=True)
            st.caption(f"score {sent['score']:+.2f} · {sent['n']} headlines")
        else:
            st.markdown('<span class="bento-big" style="color:#6b7280">N/A</span>', unsafe_allow_html=True)
            st.caption("No news / sentiment unavailable")

# --- BENTO ROW 2: per-horizon table -----------------------------------------
with st.container(border=True):
    st.markdown('<p class="bento-h">📊 Per-horizon forecast & out-of-sample backtest</p>',
                unsafe_allow_html=True)
    rows = []
    for hd in horizons:
        # News sentiment applies only as a soft ±1-notch tilt, and only when reliable.
        disp_sig = (forecast.tilt_signal(hd["signal"], sent["label"])
                    if (sent["available"] and hd["reliable"]) else hd["signal"])
        rows.append({
            "Horizon": hd["label"],
            "Predicted": format_inr(hd["predicted_price"]),
            "Exp. move": f"{hd['predicted_change']:+.2f}%",
            "MAPE": f"{hd['mape']*100:.1f}%" if hd["mape"] is not None else "N/A",
            "Directional": f"{hd['directional']*100:.0f}%" if hd["directional"] is not None else "N/A",
            "Skill vs naive": f"{hd['skill']*100:+.0f}%" if hd["skill"] is not None else "N/A",
            "Reliable": "✅" if hd["reliable"] else "⚠️ no edge",
            "Signal": disp_sig,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if sent["available"] and any(h["reliable"] for h in horizons):
        st.caption(f"Signals for reliable horizons include a soft tilt from {sent['label'].lower()} news.")

# --- BENTO ROW 3: news list + plain-English note ----------------------------
a, b = st.columns([1, 1])
with a:
    with st.container(border=True):
        st.markdown(f'<p class="bento-h">📰 Latest news — {name}</p>', unsafe_allow_html=True)
        if sent["items"]:
            tag = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}
            for art in sent["items"][:6]:
                lab = "Bullish" if art["sentiment"] > 0.05 else "Bearish" if art["sentiment"] < -0.05 else "Neutral"
                title = art["title"] or "Untitled"
                title = title[:120] + ("…" if len(title) > 120 else "")
                st.markdown(
                    f"- {tag[lab]} [{title}]({art['url']}) "
                    f"<span class='ctp-muted'>· {art['source']} {art['published']}</span>",
                    unsafe_allow_html=True)
        else:
            st.caption(f"No recent news for {name}. (Add a free API key for more — see HOW_TO_RUN.md.)")
with b:
    with st.container(border=True):
        st.markdown('<p class="bento-h">📖 What these numbers mean</p>', unsafe_allow_html=True)
        st.markdown(
            "- **MAPE** = average % the forecast was off, on **unseen** data (lower is better).\n"
            "- **Directional** = how often it got tomorrow's up/down right (50% = a coin flip).\n"
            "- **Skill vs naive** = % less error than a random-walk baseline; **>0 means it has an edge**.\n"
            "- **Longer horizons (1W, 1M) are far less reliable** — error grows fast; expect ⚠️ often.\n"
            "- A horizon only shows Buy/Sell when it's reliable **and** the expected move beats its own error."
        )
        st.caption("Daily crypto is close to a random walk; no model reliably beats the market. "
                   "⚠️ Educational only — not financial advice.")

cols = st.columns(2)
if cols[0].button("📄 Full coin details →", width="stretch"):
    ui.go_to_coin(coin_id)
cols[1].link_button("🔗 View on CoinGecko", f"https://www.coingecko.com/en/coins/{coin_id}",
                    width="stretch")
