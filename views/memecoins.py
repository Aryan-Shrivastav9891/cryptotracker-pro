"""Memecoins view — port of pages/memecoins.js (now a real, working page).

Identifies well-known meme coins by id/symbol from the live market list.
"""
import streamlit as st

from lib import coingecko, enrich, ui

dark = st.session_state.get("dark_mode", False)

# Curated set of popular meme coins (matched by CoinGecko id or symbol).
MEME_IDS = {
    "dogecoin", "shiba-inu", "pepe", "dogwifcoin", "bonk", "floki",
    "memecoin", "book-of-meme", "baby-doge-coin", "brett-based",
    "mog-coin", "popcat", "cat-in-a-dogs-world", "dogelon-mars",
}
MEME_SYMBOLS = {
    "doge", "shib", "pepe", "wif", "bonk", "floki", "meme", "bome",
    "babydoge", "brett", "mog", "popcat", "mew", "elon",
}

st.subheader("⚡ Memecoins")
st.caption("Popular community / meme tokens — high risk, high volatility.")

coins = coingecko.load_markets(per_page=250, sparkline=True)
if not coins:
    st.warning("No market data available right now.")
    st.stop()

coins, _ = enrich.enrich_coins(coins)
memes = [
    c for c in coins
    if c.get("id") in MEME_IDS or (c.get("symbol") or "").lower() in MEME_SYMBOLS
]
memes = enrich.filter_by_search(memes, st.session_state.get("search", ""))

if not memes:
    st.info("No memecoins from our list are in the current top markets.")
    st.stop()

ui.coin_grid(memes, dark, key_prefix="meme")
