"""Crypto news (CryptoCompare). No API key required."""
from __future__ import annotations

from typing import Any, Dict, List

import requests
import streamlit as st

NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
_HEADERS = {"Accept": "application/json", "User-Agent": "CryptoTrackerPro/1.0 (Streamlit)"}
_TIMEOUT = 20


@st.cache_data(ttl=600, show_spinner=False)
def get_news(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Latest news articles for a coin symbol. Cached for 10 minutes.

    Returns a list of dicts with keys: title, url, imageurl, published_on,
    source. Never raises — returns [] on any failure.
    """
    if not symbol:
        return []
    try:
        resp = requests.get(
            NEWS_URL,
            params={"categories": symbol.upper()},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("Data", []) or []
        return data[:limit]
    except requests.RequestException:
        return []
