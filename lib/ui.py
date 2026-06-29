"""Shared Streamlit UI: theme, header, sidebar controls, market overview,
coin cards and footer. Ports the React components (Header, Tabs, MarketOverview,
CoinCard, Footer) into reusable Streamlit helpers.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List

import streamlit as st

from lib import glossary, model
from lib.formatting import (
    change_color,
    format_inr,
    format_pct,
    human_format,
    safe_float,
)

COIN_DETAILS_PAGE = "views/coin_details.py"


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    st.session_state.setdefault("search", "")
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("selected_coin_id", None)
    st.session_state.setdefault("show_count", 30)


# --------------------------------------------------------------------------- #
# Theming (runtime dark mode via injected CSS)
# --------------------------------------------------------------------------- #
def apply_theme(dark: bool) -> None:
    if dark:
        css = """
        <style>
        .stApp { background-color: #0f172a; }
        .stApp, .stApp p, .stApp span, .stApp li,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #e5e7eb; }
        [data-testid="stSidebar"] { background-color: #111827; }
        .ctp-card { background-color: #1f2937; border: 1px solid #374151; }
        .ctp-muted { color: #9ca3af !important; }
        </style>
        """
    else:
        css = """
        <style>
        .ctp-card { background-color: #ffffff; border: 1px solid #e5e7eb; }
        .ctp-muted { color: #6b7280 !important; }
        </style>
        """
    # Brand styling shared by both themes.
    css += """
    <style>
    .ctp-brand {
        font-size: 2.2rem; font-weight: 800; margin-bottom: 0;
        background: linear-gradient(90deg, #7c3aed, #3b82f6);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ctp-card { border-radius: 0.85rem; padding: 1rem 1.15rem; }
    .ctp-badge { padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .ctp-pill { padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; }

    /* --- Glossary hover tooltips (work in both light & dark) --- */
    .gloss {
        border-bottom: 1px dotted #7c3aed;
        cursor: help;
        position: relative;
    }
    .gloss::after {
        content: attr(data-tip);
        position: absolute; left: 50%; bottom: 150%; transform: translateX(-50%);
        width: 240px; max-width: 80vw; white-space: normal; text-align: left;
        background: #1f2937; color: #f9fafb;
        border: 1px solid #7c3aed; border-radius: 8px;
        padding: 8px 10px; font-size: 0.78rem; font-weight: 400; line-height: 1.35;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        opacity: 0; visibility: hidden; transition: opacity .15s ease;
        z-index: 9999; pointer-events: none;
    }
    .gloss::before {
        content: ""; position: absolute; left: 50%; bottom: 150%;
        transform: translate(-50%, 100%);
        border: 6px solid transparent; border-top-color: #7c3aed;
        opacity: 0; visibility: hidden; transition: opacity .15s ease; z-index: 9999;
    }
    .gloss:hover::after, .gloss:hover::before { opacity: 1; visibility: visible; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Glossary tooltips (inline hover help for jargon)
# --------------------------------------------------------------------------- #
_terms_sorted = sorted(glossary.terms(), key=len, reverse=True)
_GLOSS_PATTERN = (
    re.compile(r"(?<![\w])(" + "|".join(re.escape(t) for t in _terms_sorted) + r")s?(?![\w])",
               re.IGNORECASE)
    if _terms_sorted else None
)


def term(label: str, definition: str = None) -> str:
    """Return an HTML span that shows ``definition`` on hover (with native title fallback).

    If ``definition`` is omitted it's looked up in the glossary. Unknown terms are
    returned as plain (escaped) text. Render with st.markdown(..., unsafe_allow_html=True).
    """
    if definition is None:
        definition = glossary.get_definition(label)
    lab = html.escape(str(label))
    if not definition:
        return lab
    tip = html.escape(definition, quote=True)
    return f'<span class="gloss" data-tip="{tip}" title="{tip}">{lab}</span>'


def _wrap_match(match: "re.Match", used: set) -> str:
    matched = match.group(0)
    definition = glossary.get_definition(matched)
    if not definition:
        return matched
    key = glossary.normalize(matched).rstrip("s")
    if key in used:  # only the first occurrence of each term per text block
        return matched
    used.add(key)
    return term(matched, definition)


def annotate(text: str) -> str:
    """Auto-wrap the first occurrence of each known glossary term in a hover tooltip.

    Skips text inside `backtick code spans`, never double-wraps, and matches
    whole words case-insensitively. Render with st.markdown(..., unsafe_allow_html=True).
    """
    if not text or _GLOSS_PATTERN is None:
        return text
    used: set = set()
    out: List[str] = []
    for part in re.split(r"(`[^`]*`)", str(text)):  # keep code spans verbatim
        if part[:1] == "`":
            out.append(part)
        else:
            out.append(_GLOSS_PATTERN.sub(lambda m: _wrap_match(m, used), part))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Header / sidebar
# --------------------------------------------------------------------------- #
def top_bar() -> None:
    st.markdown('<p class="ctp-brand">CryptoTracker Pro</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="ctp-muted">Track crypto prices, AI predictions and market '
        "insights — now powered by Python.</p>",
        unsafe_allow_html=True,
    )


def sidebar_controls() -> None:
    """Render the persistent sidebar (search + dark mode + model + refresh)."""
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.text_input("🔎 Search coins", key="search", placeholder="Bitcoin, ETH…")
        st.toggle("🌙 Dark mode", key="dark_mode")
        st.divider()
        st.caption(model.model_status())
        if st.button("🔄 Refresh live data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data: CoinGecko · News: CryptoCompare")


# --------------------------------------------------------------------------- #
# Navigation helper
# --------------------------------------------------------------------------- #
def go_to_coin(coin_id: str) -> None:
    st.session_state.selected_coin_id = coin_id
    st.switch_page(COIN_DETAILS_PAGE)


# --------------------------------------------------------------------------- #
# Market overview (ports MarketOverview.js)
# --------------------------------------------------------------------------- #
def market_overview(coins: List[Dict[str, Any]]) -> None:
    if not coins:
        return
    total_mc = sum(safe_float(c.get("market_cap")) for c in coins)
    total_vol = sum(safe_float(c.get("total_volume")) for c in coins)
    changes = [safe_float(c.get("price_change_percentage_24h")) for c in coins]
    avg_change = sum(changes) / len(changes) if changes else 0.0
    bullish = avg_change > 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Status", "Bullish 🐂" if bullish else "Bearish 🐻")
    c2.metric("Avg 24h Change", format_pct(avg_change), delta=f"{avg_change:.2f}%")
    c3.metric("Total Market Cap", human_format(total_mc, prefix="₹"))
    c4.metric("24h Volume", human_format(total_vol, prefix="₹"))

    movers = sorted(
        coins,
        key=lambda c: abs(safe_float(c.get("price_change_percentage_24h"))),
        reverse=True,
    )[:3]
    if movers:
        chips = "  ".join(
            f"**{m.get('symbol', '').upper()}** "
            f"{format_pct(m.get('price_change_percentage_24h'), signed=True)}"
            for m in movers
        )
        st.caption(f"🔥 Top movers: {chips}")


# --------------------------------------------------------------------------- #
# Coin card (ports CoinCard.js)
# --------------------------------------------------------------------------- #
def _sell_advice(change_24h: float) -> Dict[str, str]:
    if change_24h < -10:
        return {
            "label": "⚠️ Sell Alert",
            "text": "Price dropped significantly in the last 24h — consider selling.",
            "bg": "rgba(239,68,68,0.12)",
            "fg": "#ef4444",
        }
    if change_24h > 20:
        return {
            "label": "💰 Profit Opportunity",
            "text": "Price rose significantly — consider locking in profits.",
            "bg": "rgba(234,179,8,0.14)",
            "fg": "#ca8a04",
        }
    return {
        "label": "📈 Market Insight",
        "text": "No immediate sell signal. Monitor the market for changes.",
        "bg": "rgba(59,130,246,0.12)",
        "fg": "#3b82f6",
    }


def coin_card(coin: Dict[str, Any], dark: bool, key: str) -> None:
    change = safe_float(coin.get("price_change_percentage_24h"))
    color = change_color(change, dark)
    arrow = "▲" if change >= 0 else "▼"
    advice = _sell_advice(change)
    rank = coin.get("market_cap_rank") or "N/A"

    with st.container(border=True):
        head = st.columns([1, 4, 2])
        with head[0]:
            if coin.get("image"):
                st.image(coin["image"], width=40)
        with head[1]:
            star = " ⭐" if (coin.get("market_cap_rank") or 999) <= 10 else ""
            st.markdown(
                f"**{coin.get('name', '?')}**{star}  \n"
                f"<span class='ctp-muted'>{coin.get('symbol', '').upper()} · "
                f"Rank #{rank}</span>",
                unsafe_allow_html=True,
            )
        with head[2]:
            st.markdown(
                f"<div style='text-align:right'>"
                f"<span class='ctp-badge' style='background:{advice['bg']};color:{color}'>"
                f"{arrow} {abs(change):.2f}%</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='font-size:1.6rem;font-weight:800'>"
            f"{format_inr(coin.get('current_price'))}</div>",
            unsafe_allow_html=True,
        )

        if coin.get("predicted_price") is not None:
            pred_pct = safe_float(coin.get("predicted_change"))
            st.markdown(
                f"<span class='ctp-muted'>AI predicted: </span>"
                f"<b>{format_inr(coin.get('predicted_price'))}</b> "
                f"<span style='color:{change_color(pred_pct, dark)}'>"
                f"({format_pct(pred_pct, signed=True)})</span>",
                unsafe_allow_html=True,
            )

        if coin.get("description"):
            desc = str(coin["description"])
            preview = desc[:100] + ("…" if len(desc) > 100 else "")
            st.markdown(
                f"<div class='ctp-muted' style='font-size:0.84rem;margin-top:0.3rem'>"
                f"{preview}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='ctp-card' style='background:{advice['bg']};"
            f"border:none;padding:0.6rem 0.8rem;margin-top:0.4rem'>"
            f"<b style='color:{advice['fg']}'>{advice['label']}</b><br>"
            f"<span style='font-size:0.82rem'>{advice['text']}</span></div>",
            unsafe_allow_html=True,
        )

        if st.button("View details →", key=key, width="stretch"):
            go_to_coin(coin.get("id"))


def coin_grid(coins: List[Dict[str, Any]], dark: bool, key_prefix: str, columns: int = 3) -> None:
    if not coins:
        st.info("No coins match your criteria. Try adjusting your search.")
        return
    cols = st.columns(columns)
    for i, coin in enumerate(coins):
        with cols[i % columns]:
            coin_card(coin, dark, key=f"{key_prefix}_{coin.get('id', i)}")


# --------------------------------------------------------------------------- #
# Footer (ports Footer.js)
# --------------------------------------------------------------------------- #
def footer() -> None:
    st.divider()
    st.markdown(
        "<div class='ctp-muted' style='text-align:center;font-size:0.85rem'>"
        "<b>CryptoTracker Pro</b> · Track crypto prices, get market insights, "
        "and make better trading decisions.<br>"
        "© 2026 CryptoTracker Pro. All rights reserved. · For educational use only — "
        "not financial advice.</div>",
        unsafe_allow_html=True,
    )
