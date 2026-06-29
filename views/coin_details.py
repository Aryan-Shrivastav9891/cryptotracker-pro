"""Coin Details view — port of pages/coin.js + components/CoinDetails.js.

Full detail for a single coin: price chart, market data, supply, multi-currency
prices, all-time highs/lows, community + developer stats, links and description.
"""
import streamlit as st

from lib import charts, coingecko
from lib.formatting import (
    format_date,
    format_inr,
    format_inr_compact,
    format_pct,
    human_format,
    safe_float,
    strip_html,
)

dark = st.session_state.get("dark_mode", False)


def dget(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# --- Coin selection ---------------------------------------------------------
coin_id = st.session_state.get("selected_coin_id")
if not coin_id:
    st.subheader("ℹ️ Coin Details")
    st.caption("Pick a coin to see full details, or click ‘View details’ from any card.")
    market = coingecko.load_markets(per_page=250)
    if not market:
        st.stop()
    options = {f"{c.get('name')} ({(c.get('symbol') or '').upper()})": c.get("id") for c in market}
    choice = st.selectbox("Choose a coin", list(options.keys()))
    coin_id = options.get(choice)
    st.session_state.selected_coin_id = coin_id

coin = coingecko.load_coin(coin_id)
if not coin:
    st.error("Could not load this coin. It may not exist or the API is unavailable.")
    st.stop()

md = coin.get("market_data") or {}
symbol = (coin.get("symbol") or "").upper()
change_24h = safe_float(md.get("price_change_percentage_24h"))
positive = change_24h >= 0

# --- Header -----------------------------------------------------------------
back, _ = st.columns([1, 5])
with back:
    if st.button("← Back to list"):
        st.session_state.selected_coin_id = None
        st.switch_page("views/home.py")

h1, h2 = st.columns([3, 2])
with h1:
    img = dget(coin, "image", "large")
    cols = st.columns([1, 5])
    if img:
        cols[0].image(img, width=56)
    cols[1].markdown(
        f"## {coin.get('name')} <span class='ctp-muted'>({symbol})</span>",
        unsafe_allow_html=True,
    )
    rank = coin.get("market_cap_rank")
    if rank:
        st.caption(f"Rank #{rank}")
with h2:
    st.metric(
        "Price (INR)",
        format_inr(dget(md, "current_price", "inr")),
        delta=f"{change_24h:.2f}% (24h)",
    )
    st.markdown(
        f"[View on CoinGecko →](https://www.coingecko.com/en/coins/{coin.get('id')})"
    )

st.divider()

# --- Price chart ------------------------------------------------------------
st.markdown("### 📈 7-Day Price Trend")
prices = dget(md, "sparkline_7d", "price", default=[]) or []
prices = [float(p) for p in prices if p is not None]
if prices:
    fig = charts.line_chart(prices, positive=positive, dark=dark, height=380)
    st.plotly_chart(fig, use_container_width=True, key="detail_chart")
else:
    st.info("No price history available for this coin.")

# --- Market data ------------------------------------------------------------
st.markdown("### 💹 Market Data")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Market Cap (INR)", format_inr_compact(dget(md, "market_cap", "inr")))
    st.caption(f"$ {human_format(dget(md, 'market_cap', 'usd'))}")
    mc_chg = safe_float(md.get("market_cap_change_percentage_24h"))
    st.caption(f"{format_pct(mc_chg, signed=True)} (24h)")
with c2:
    st.metric("24h Volume (INR)", format_inr_compact(dget(md, "total_volume", "inr")))
    st.caption(f"$ {human_format(dget(md, 'total_volume', 'usd'))}")
with c3:
    st.metric("Circulating Supply", human_format(md.get("circulating_supply")))
    st.caption(f"Total: {human_format(md.get('total_supply'))}")
    st.caption(f"Max: {human_format(md.get('max_supply')) if md.get('max_supply') else 'Unlimited'}")

# --- Prices & all-time stats ------------------------------------------------
p1, p2 = st.columns(2)
with p1:
    st.markdown("#### 💱 Price Information")
    st.write(
        {
            "INR": format_inr(dget(md, "current_price", "inr")),
            "USD": "$ " + human_format(dget(md, "current_price", "usd")),
            "EUR": "€ " + human_format(dget(md, "current_price", "eur")),
            "BTC": str(dget(md, "current_price", "btc", default="N/A")),
            "24h Change": format_pct(md.get("price_change_percentage_24h"), signed=True),
            "7d Change": format_pct(md.get("price_change_percentage_7d"), signed=True),
        }
    )
with p2:
    st.markdown("#### 🏔️ All-Time Stats")
    ath = dget(md, "ath", "inr")
    atl = dget(md, "atl", "inr")
    ath_date = format_date(dget(md, "ath_date", "inr"))
    atl_date = format_date(dget(md, "atl_date", "inr"))
    st.write(
        {
            "All-Time High (INR)": format_inr(ath),
            "ATH Date": ath_date or "N/A",
            "ATH Change": format_pct(dget(md, "ath_change_percentage", "inr"), signed=True),
            "All-Time Low (INR)": format_inr(atl),
            "ATL Date": atl_date or "N/A",
            "ATL Change": format_pct(dget(md, "atl_change_percentage", "inr"), signed=True),
        }
    )

# --- Community & developer stats -------------------------------------------
cd = coin.get("community_data") or {}
dd = coin.get("developer_data") or {}
s1, s2 = st.columns(2)
with s1:
    st.markdown("#### 👥 Community Stats")
    st.write(
        {
            "Twitter Followers": human_format(cd.get("twitter_followers")),
            "Reddit Subscribers": human_format(cd.get("reddit_subscribers")),
            "Telegram Users": human_format(cd.get("telegram_channel_user_count")),
            "Reddit Active": human_format(cd.get("reddit_accounts_active_48h")),
        }
    )
with s2:
    st.markdown("#### 🛠️ Developer Activity")
    st.write(
        {
            "GitHub Stars": human_format(dd.get("stars")),
            "GitHub Forks": human_format(dd.get("forks")),
            "Subscribers": human_format(dd.get("subscribers")),
            "Total Issues": human_format(dd.get("total_issues")),
            "Commits (4 weeks)": human_format(dd.get("commit_count_4_weeks")),
        }
    )

# --- Links ------------------------------------------------------------------
links = coin.get("links") or {}
link_items = []
homepage = next((u for u in (links.get("homepage") or []) if u), None)
if homepage:
    link_items.append(f"[🌐 Website]({homepage})")
if links.get("subreddit_url"):
    link_items.append(f"[👽 Reddit]({links['subreddit_url']})")
if links.get("twitter_screen_name"):
    link_items.append(f"[🐦 Twitter](https://twitter.com/{links['twitter_screen_name']})")
github = next((u for u in dget(links, "repos_url", "github", default=[]) or [] if u), None)
if github:
    link_items.append(f"[💻 GitHub]({github})")
if link_items:
    st.markdown("#### 🔗 Links")
    st.markdown(" · ".join(link_items))

# --- Description ------------------------------------------------------------
# Sanitize: CoinGecko descriptions are community-editable; strip HTML (no XSS).
description = strip_html(dget(coin, "description", "en"))
if description:
    st.markdown(f"#### 📄 About {coin.get('name')}")
    st.markdown(description)
