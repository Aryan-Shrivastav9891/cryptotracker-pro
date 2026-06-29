"""Profitable view — port of pages/profitable.js.

Shows coins up more than 10% in the last 24 hours, as both a quick table and a
card grid.
"""
import pandas as pd
import streamlit as st

from lib import coingecko, enrich, ui
from lib.formatting import format_inr, format_pct

dark = st.session_state.get("dark_mode", False)

st.subheader("📊 Profitable Coins")
st.caption("Coins with more than +10% price change in the last 24 hours.")

coins = coingecko.load_markets(per_page=250, sparkline=True)
if not coins:
    st.warning("No market data available right now.")
    st.stop()

coins, _ = enrich.enrich_coins(coins)
profitable = [c for c in coins if (c.get("price_change_percentage_24h") or 0) > 10]
profitable = enrich.filter_by_search(profitable, st.session_state.get("search", ""))
profitable.sort(key=lambda c: c.get("price_change_percentage_24h") or 0, reverse=True)

if not profitable:
    st.info("No coins are up more than 10% right now. Check back later.")
    st.stop()

table = pd.DataFrame(
    [
        {
            "Rank": c.get("market_cap_rank"),
            "Name": c.get("name"),
            "Symbol": (c.get("symbol") or "").upper(),
            "Price": format_inr(c.get("current_price")),
            "24h Change": format_pct(c.get("price_change_percentage_24h"), signed=True),
        }
        for c in profitable
    ]
)
st.dataframe(table, width="stretch", hide_index=True)

st.divider()
ui.coin_grid(profitable, dark, key_prefix="profit")
