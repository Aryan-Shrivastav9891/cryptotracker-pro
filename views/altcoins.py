"""Altcoins view — port of pages/altcoins.js (now a real, working page).

Altcoins = everything outside the top-10 by market cap.
"""
import streamlit as st

from lib import coingecko, enrich, ui

dark = st.session_state.get("dark_mode", False)

st.subheader("⭐ Altcoins")
st.caption("Coins ranked outside the top 10 by market capitalisation.")

coins = coingecko.load_markets(per_page=250, sparkline=True)
if not coins:
    st.warning("No market data available right now.")
    st.stop()

coins, _ = enrich.enrich_coins(coins)
altcoins = [c for c in coins if (c.get("market_cap_rank") or 9999) > 10]
altcoins = enrich.filter_by_search(altcoins, st.session_state.get("search", ""))

if not altcoins:
    st.info("No altcoins match your search.")
    st.stop()

max_n = min(120, len(altcoins))
n = st.slider("Coins to show", 6, max_n, min(30, max_n), step=6) if max_n > 6 else max_n
ui.coin_grid(altcoins[:n], dark, key_prefix="alt")
