"""Home / "All" view — port of pages/index.js.

Loads the market list, runs the AI prediction model, shows the market overview
and a searchable, paginated grid of coin cards with predicted prices.
"""
import streamlit as st

from lib import coingecko, enrich, ui

dark = st.session_state.get("dark_mode", False)

coins = coingecko.load_markets(per_page=250, sparkline=True)
if not coins:
    st.warning("No market data available right now. Try the 🔄 Refresh button shortly.")
    st.stop()

coins, engine = enrich.enrich_coins(coins)

ui.market_overview(coins[:50])
st.caption(f"🤖 Prediction engine: {engine}")
st.divider()

filtered = enrich.filter_by_search(coins, st.session_state.get("search", ""))

st.subheader(f"All Coins · {len(filtered)} match")
if not filtered:
    st.info("No coins match your search. Try a different term.")
    st.stop()

max_n = min(120, len(filtered))
if max_n > 6:
    n = st.slider("Coins to show", min_value=6, max_value=max_n,
                  value=min(30, max_n), step=6)
else:
    n = max_n

ui.coin_grid(filtered[:n], dark, key_prefix="home")
