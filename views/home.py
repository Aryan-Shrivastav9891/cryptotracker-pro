"""Home / "All" view — port of pages/index.js.

Loads the market list, runs the AI prediction model, shows the market overview
and a searchable, paginated grid of coin cards with predicted prices.
"""
import streamlit as st

from lib import coingecko, enrich, glossary, ui
from lib.formatting import safe_float

dark = st.session_state.get("dark_mode", False)

coins = coingecko.load_markets(per_page=250, sparkline=True)
if not coins:
    st.warning("No market data available right now. Try the 🔄 Refresh button shortly.")
    st.stop()

coins, engine = enrich.enrich_coins(coins)

# --- Animated bento hero (market overview, anime.js) ------------------------
_total_mc = sum(safe_float(c.get("market_cap")) for c in coins)
_total_vol = sum(safe_float(c.get("total_volume")) for c in coins)
_changes = [safe_float(c.get("price_change_percentage_24h")) for c in coins]
_avg = sum(_changes) / len(_changes) if _changes else 0.0
_top = max(coins, key=lambda c: safe_float(c.get("price_change_percentage_24h")), default=None)
_spark = [float(p) for p in ((coins[0].get("sparkline_in_7d") or {}).get("price") or []) if p is not None]
ui.animated_bento(
    dark,
    hero={"label": f"Total Market Cap · top {len(coins)}", "value": _total_mc,
          "fmt": "compact", "decimals": 2, "delta": _avg, "spark": _spark[-40:]},
    kpis=[
        {"label": "24h Volume", "value": _total_vol, "fmt": "compact", "decimals": 2,
         "tip": glossary.get_definition("Volume (24h)")},
        {"label": "Avg 24h Change", "value": _avg, "fmt": "pct", "decimals": 2, "signed": True},
        {"label": "Coins Tracked", "value": len(coins), "fmt": "num", "decimals": 0},
        {"label": "Top Gainer 24h", "value": safe_float(_top.get("price_change_percentage_24h")) if _top else 0.0,
         "fmt": "pct", "decimals": 1, "signed": True},
    ],
    signal={"text": "Bullish" if _avg > 0 else "Bearish", "kind": "buy" if _avg > 0 else "sell"},
    height=250,
)
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
