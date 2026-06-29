"""Future Gains — AI Forecast Lab (ensemble, backtested, honest).

Bentobox layout: a grid of tiles for the forecast chart + 80% band, the signal
with its reasoning and reliability, backtest metrics (skill vs a naive baseline),
the per-model leaderboard, and news + sentiment. The page never shows a confident
number the out-of-sample backtest can't justify.
"""
import pandas as pd
import streamlit as st

from lib import charts, coingecko, forecast, news, ui
from lib.formatting import format_inr, format_pct, safe_float

dark = st.session_state.get("dark_mode", False)

# --- bento styling ----------------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        height: 100%;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.25rem 0.4rem; }
    .bento-h { font-size:0.8rem; text-transform:uppercase; letter-spacing:.04em; opacity:.7; margin:0; }
    .bento-big { font-size:2rem; font-weight:800; line-height:1.1; margin:.1rem 0; }
    .bento-pill { display:inline-block; padding:3px 12px; border-radius:999px;
                  font-weight:700; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SIG_COLORS = {"Strong Buy": "#16a34a", "Buy": "#22c55e", "Hold": "#3b82f6",
              "Sell": "#f97316", "Strong Sell": "#ef4444"}

st.markdown("## ⏳ AI Forecast Lab")
st.caption("Ensemble forecast, **backtested on unseen data**. It beats a naive baseline — or it tells you it doesn't.")

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
    ctrl = st.columns([3, 1, 1])
    chosen_label = ctrl[0].selectbox("Coin", label_list, index=default_index)
    coin_id = labels[chosen_label]
    horizon = ctrl[1].selectbox("Horizon", [7, 14, 30], index=0, format_func=lambda d: f"{d} days")
    days = ctrl[2].selectbox("History", [365, 540, 730], index=1, format_func=lambda d: f"{d} days")

coin = by_id.get(coin_id, {})
name = coin.get("name", coin_id)
symbol = (coin.get("symbol") or "").upper()

# --- train + backtest + forecast + sentiment --------------------------------
with st.status(f"Training ensemble for {name}…", expanded=True) as status:
    st.write(f"📥 Loading ~{days} days of daily prices (Binance → CoinGecko fallback)…")
    st.write("🧪 Walk-forward backtest (one-step, refit each day) vs a naive baseline…")
    result = forecast.forecast_coin(coin_id, symbol, days=days, horizon=horizon)
    if not result:
        status.update(label="Not enough history to backtest", state="error")
        st.warning(f"Not enough daily history to backtest {name} honestly. Try another coin.")
        st.stop()
    st.write("📰 Fetching news + scoring sentiment…")
    articles = news.get_news(symbol, limit=8)
    sentiment = news.analyze_sentiment(articles)
    status.update(label=f"Done · {result['engine']} · data: {result['source']}", state="complete")

# --- derive -----------------------------------------------------------------
pct = safe_float(result["predicted_change"])
positive = pct >= 0
reliable = result["reliable"]
base_signal = result["signal"]
ens = result["ensemble"]
mape = ens["mape"]
directional = ens["directional"]
skill = result["skill"]

mape_txt = f"{mape * 100:.1f}%" if mape is not None else "N/A"
dir_txt = f"{directional * 100:.0f}%" if directional is not None else "N/A"
skill_txt = f"{skill * 100:+.0f}%" if skill is not None else "N/A"

# Sentiment is a SOFT tilt only, and only when the forecast is reliable.
tilted_signal = forecast.tilt_signal(base_signal, sentiment["label"]) if (sentiment["available"] and reliable) else base_signal
display_signal = tilted_signal
sig_color = SIG_COLORS.get(display_signal, "#6b7280")

# =========================================================================== #
# BENTO ROW 1 — forecast chart (wide)  +  signal & sentiment (stacked)
# =========================================================================== #
left, right = st.columns([2, 1])
with left:
    with st.container(border=True):
        st.markdown('<p class="bento-h">🔮 Forecast & 80% confidence band</p>', unsafe_allow_html=True)
        fig = charts.forecast_interval_chart(
            result["history_dates"], result["history_prices"],
            result["forecast_dates"], result["forecast_prices"],
            result["lower"], result["upper"], positive=positive, dark=dark, height=360,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"fc_{coin_id}_{horizon}_{days}")

with right:
    with st.container(border=True):
        st.markdown('<p class="bento-h">Signal</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="bento-pill" style="background:{sig_color}22;color:{sig_color}">'
            f'{display_signal}</span>', unsafe_allow_html=True)
        if reliable:
            st.markdown('<p style="color:#16a34a;font-weight:600;margin:.4rem 0 0">✅ Has edge vs naive</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#d97706;font-weight:600;margin:.4rem 0 0">⚠️ No measured edge</p>',
                        unsafe_allow_html=True)
        st.caption(result["reasoning"])
        if sentiment["available"] and reliable and tilted_signal != base_signal:
            st.caption(f"(soft tilt from {sentiment['label'].lower()} news; base: {base_signal})")

    with st.container(border=True):
        st.markdown('<p class="bento-h">📰 News sentiment</p>', unsafe_allow_html=True)
        if sentiment["available"]:
            s_color = {"Bullish": "#16a34a", "Bearish": "#ef4444"}.get(sentiment["label"], "#3b82f6")
            st.markdown(
                f'<span class="bento-big" style="color:{s_color}">{sentiment["label"]}</span>',
                unsafe_allow_html=True)
            st.caption(f"score {sentiment['score']:+.2f} · "
                       f"🟢 {sentiment['bullish']}  ⚪ {sentiment['neutral']}  🔴 {sentiment['bearish']}")
        else:
            st.markdown('<span class="bento-big" style="color:#6b7280">N/A</span>', unsafe_allow_html=True)
            st.caption("No news / sentiment unavailable")

# =========================================================================== #
# BENTO ROW 2 — three metric tiles
# =========================================================================== #
g = st.columns(3)
with g[0]:
    with st.container(border=True):
        st.markdown('<p class="bento-h">Skill vs naive</p>', unsafe_allow_html=True)
        beat_color = "#16a34a" if (skill is not None and skill > 0) else "#ef4444"
        st.markdown(f'<span class="bento-big" style="color:{beat_color}">{skill_txt}</span>',
                    unsafe_allow_html=True)
        st.caption("lower error than a random-walk baseline" if (skill is not None and skill > 0)
                   else "does not beat the naive baseline")
with g[1]:
    with st.container(border=True):
        st.markdown('<p class="bento-h">Backtest (out-of-sample)</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="bento-big">{mape_txt}</span>', unsafe_allow_html=True)
        st.caption(f"avg error (MAPE) · directional {dir_txt}")
with g[2]:
    with st.container(border=True):
        st.markdown('<p class="bento-h">Forecast</p>', unsafe_allow_html=True)
        c_color = "#16a34a" if positive else "#ef4444"
        st.markdown(f'<span class="bento-big" style="color:{c_color}">{pct:+.2f}%</span>',
                    unsafe_allow_html=True)
        st.caption(f"{format_inr(result['current_price'])} → {format_inr(result['predicted_price'])} "
                   f"(+{horizon}d)")

# =========================================================================== #
# BENTO ROW 3 — model leaderboard  +  latest news
# =========================================================================== #
a, b = st.columns([1, 1])
with a:
    with st.container(border=True):
        st.markdown('<p class="bento-h">🏁 Model leaderboard (backtest)</p>', unsafe_allow_html=True)
        rows = []
        for m in sorted(result["models"], key=lambda x: (x["mape"] if x["mape"] is not None else 9)):
            rows.append({
                "Model": m["name"],
                "MAPE": f"{m['mape']*100:.2f}%" if m["mape"] is not None else "N/A",
                "Dir": f"{m['directional']*100:.0f}%" if m["directional"] is not None else "N/A",
                "Weight": f"{m['weight']*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
with b:
    with st.container(border=True):
        st.markdown(f'<p class="bento-h">📰 Latest news — {name}</p>', unsafe_allow_html=True)
        if articles:
            from datetime import datetime, timezone
            for art in articles[:5]:
                ts = art.get("published_on")
                when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
                st.markdown(
                    f"- [{art.get('title', 'Untitled')}]({art.get('url', '#')}) "
                    f"<span class='ctp-muted'>· {art.get('source', '')} {when}</span>",
                    unsafe_allow_html=True)
        else:
            st.caption(f"No recent news for {name}.")

# =========================================================================== #
# BENTO ROW 4 — plain-English explainer + actions
# =========================================================================== #
with st.container(border=True):
    st.markdown('<p class="bento-h">📖 What these numbers mean</p>', unsafe_allow_html=True)
    edge = ("**beats** a naive random-walk baseline" if (skill is not None and skill > 0)
            else "does **not** beat a naive baseline")
    st.markdown(
        f"Over the last **{result['lookback']} days** of *unseen* data, the ensemble's average "
        f"error was **{mape_txt}** and it called the next day's direction **{dir_txt}** of the time — "
        f"so it {edge} (**skill {skill_txt}**). "
        f"The forecast projects **{pct:+.2f}%** over the next **{horizon} days**, but the **80% band** shows "
        f"how wide the realistic range is. "
        + ("Because there's no measured edge, the signal stays **Hold** — that's the honest call."
           if not reliable else
           "Treat the signal as one input, not a certainty.")
    )
    st.caption("Daily crypto is close to a random walk; no model reliably beats the market. "
               "⚠️ Educational only — not financial advice.")

cols = st.columns(2)
if cols[0].button("📄 Full coin details →", width="stretch"):
    ui.go_to_coin(coin_id)
cols[1].link_button("🔗 View on CoinGecko", f"https://www.coingecko.com/en/coins/{coin_id}",
                    width="stretch")
