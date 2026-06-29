"""Future Gains view — real-world AI price forecasting (rebuilt).

Replaces the original toy model with an honest forecasting workflow for a single
coin: fetch real historical daily prices -> walk-forward backtest (measure error
on unseen data) -> forecast ahead with a confidence band -> recommend, grounded
in the measured accuracy. Powered by statsmodels (Holt damped-trend), with a
random-walk-drift fallback.

Not financial advice — markets are largely unpredictable; this shows a backtested
estimate together with how wrong it has been.
"""
import streamlit as st

from lib import charts, coingecko, forecast, news, ui
from lib.formatting import change_color, format_inr, format_pct, safe_float

dark = st.session_state.get("dark_mode", False)

st.subheader("⏳ AI-Powered Future Forecast")
st.caption(
    "Real workflow: train on historical prices → backtest on unseen data → "
    "forecast ahead with a confidence band. Honest, not magic."
)

# --- Controls ---------------------------------------------------------------
market = coingecko.load_markets(per_page=100, sparkline=False)
if not market:
    st.warning("No market data available right now.")
    st.stop()

by_id = {c.get("id"): c for c in market}
labels = {f"{c.get('name')} ({(c.get('symbol') or '').upper()})": c.get("id") for c in market}
label_list = list(labels.keys())

# Default to the coin chosen elsewhere (a card), else the top coin.
preselected = st.session_state.get("selected_coin_id")
default_index = 0
if preselected in by_id:
    for i, lbl in enumerate(label_list):
        if labels[lbl] == preselected:
            default_index = i
            break

ctrl = st.columns([3, 1, 1])
chosen_label = ctrl[0].selectbox("Coin", label_list, index=default_index)
coin_id = labels[chosen_label]
horizon = ctrl[1].selectbox("Forecast horizon", [7, 14, 30], index=0,
                            format_func=lambda d: f"{d} days")
days = ctrl[2].selectbox("History window", [90, 180, 365], index=1,
                         format_func=lambda d: f"{d} days")

coin = by_id.get(coin_id, {})
name = coin.get("name", coin_id)
symbol = (coin.get("symbol") or "").upper()

# --- Train + backtest + forecast (with a real progress panel) ---------------
with st.status(f"Training forecast model for {name}…", expanded=True) as status:
    st.write(f"📥 Fetching {days} days of real historical prices from CoinGecko…")
    st.write("🧪 Walk-forward backtest on held-out data (data the model never saw)…")
    result = forecast.forecast_coin(coin_id, days=days, horizon=horizon)
    if not result:
        status.update(label="Not enough history to forecast", state="error")
        st.warning(f"Not enough historical data to forecast {name}. Try another coin.")
        st.stop()

    mape = result["mape"]
    directional = result["directional"]
    mape_txt = f"{mape * 100:.1f}%" if mape is not None else "N/A"
    dir_txt = f"{directional * 100:.0f}%" if directional is not None else "N/A"
    st.write(f"📊 Backtest over {result['folds']} folds → MAPE {mape_txt} · directional {dir_txt}")
    st.write(f"🔮 Forecasting {horizon} days ahead with an 80% confidence band…")
    status.update(label=f"Model trained on {result['days']} days · {result['engine']}",
                  state="complete")

pct = safe_float(result["predicted_change"])
positive = pct >= 0
confidence = result["confidence"]
rec = result["recommendation"]

# --- Headline metrics -------------------------------------------------------
m = st.columns(4)
m[0].metric("Current price", format_inr(result["current_price"]))
m[1].metric(f"Forecast (+{horizon}d)", format_inr(result["predicted_price"]),
            delta=f"{pct:.2f}%")
m[2].metric("Confidence", f"{confidence:.0f}%")
m[3].metric("Signal", rec)

# --- Backtest accuracy (the honesty panel) ----------------------------------
st.markdown("##### 🧪 Backtest accuracy (on unseen data)")
b = st.columns(3)
b[0].metric("MAPE (avg % error)", mape_txt)
b[1].metric("Directional accuracy", dir_txt)
b[2].metric("MAE (avg ₹ error)", format_inr(result["mae"]) if result["mae"] is not None else "N/A")

# --- Forecast chart ---------------------------------------------------------
st.markdown("##### 🔮 Forecast")
fig = charts.forecast_interval_chart(
    result["history_dates"], result["history_prices"],
    result["forecast_dates"], result["forecast_prices"],
    result["lower"], result["upper"],
    positive=positive, dark=dark,
)
st.plotly_chart(fig, use_container_width=True, key=f"forecast_{coin_id}_{horizon}_{days}")

# --- Honest analysis --------------------------------------------------------
rec_color = {"Strong Buy": "#16a34a", "Buy": "#22c55e", "Hold": "#3b82f6",
             "Watch": "#eab308", "Sell": "#ef4444"}.get(rec, "#6b7280")

if confidence < 35 or (mape is not None and mape > 0.25):
    quality = ("The backtest error is high, so treat this forecast as low-confidence — "
               "crypto over this horizon behaves close to a random walk.")
elif confidence > 60:
    quality = "The model tracked unseen data reasonably well, so this estimate is comparatively reliable."
else:
    quality = "Moderate backtest accuracy — use this as one input, not a certainty."

direction = (f"projects **{name}** to move **{pct:+.2f}%** over the next {horizon} days "
             f"(≈ {format_inr(result['predicted_price'])})")
st.markdown(
    f"**AI analysis:** The model {direction}. "
    f"<span style='color:{rec_color};font-weight:700'>Signal: {rec}.</span> {quality}",
    unsafe_allow_html=True,
)

if result.get("band_clipped"):
    st.warning(
        "⚠️ The lower confidence bound hit the ₹0 price floor, so the 80% band is "
        "asymmetric and its real coverage is below 80% — common for very low-priced "
        "or sharply declining coins."
    )

st.caption(
    "ℹ️ Even at higher confidence, short-horizon crypto is close to a random walk — "
    "treat this as exploratory analysis, not a trading signal."
)

# --- News -------------------------------------------------------------------
with st.expander(f"📰 Latest news for {name}"):
    articles = news.get_news(symbol, limit=6)
    if articles:
        from datetime import datetime, timezone
        for art in articles:
            ts = art.get("published_on")
            when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
            st.markdown(
                f"- [{art.get('title', 'Untitled')}]({art.get('url', '#')}) "
                f"<span class='ctp-muted'>· {art.get('source', '')} {when}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption(f"No recent news available for {name}.")

cols = st.columns(2)
if cols[0].button("📄 Full coin details →", width="stretch"):
    ui.go_to_coin(coin_id)
cols[1].link_button("🔗 View on CoinGecko", f"https://www.coingecko.com/en/coins/{coin_id}",
                    width="stretch")

st.caption("⚠️ For educational use only. Forecasts are uncertain and not financial advice.")
