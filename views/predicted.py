"""AI Predicted view — port of pages/predicted.js (now a real, working page).

Ranks coins by the model's predicted upside and highlights the strongest
predicted gainers.
"""
import streamlit as st

from lib import coingecko, enrich, ui

dark = st.session_state.get("dark_mode", False)

st.subheader("📈 AI Predicted")
st.caption("Coins ranked by the prediction model's projected upside.")

coins = coingecko.load_markets(per_page=250, sparkline=True)
if not coins:
    st.warning("No market data available right now.")
    st.stop()

coins, engine = enrich.enrich_coins(coins)
st.caption(f"🤖 Prediction engine: {engine}")

threshold = st.slider("Minimum predicted upside (%)", 0, 50, 5, step=1)
predicted = [c for c in coins if (c.get("predicted_change") or 0) >= threshold]
predicted = enrich.filter_by_search(predicted, st.session_state.get("search", ""))
predicted.sort(key=lambda c: c.get("predicted_change") or 0, reverse=True)

st.markdown(f"**{len(predicted)}** coins predicted to gain at least **{threshold}%**.")
if not predicted:
    st.info("No coins meet that predicted-upside threshold. Lower the slider.")
    st.stop()

max_n = min(60, len(predicted))
n = st.slider("Coins to show", 6, max_n, min(24, max_n), step=6) if max_n > 6 else max_n
ui.coin_grid(predicted[:n], dark, key_prefix="pred")
