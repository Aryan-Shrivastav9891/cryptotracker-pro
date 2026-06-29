"""CoinGecko data access — the Python equivalent of the original axios calls.

All network calls are wrapped in Streamlit's data cache so the free CoinGecko
endpoints are not hammered (which would trigger HTTP 429 rate-limiting and make
the app feel anything but "smooth").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
import streamlit as st

BASE_URL = "https://api.coingecko.com/api/v3"
_HEADERS = {"Accept": "application/json", "User-Agent": "CryptoTrackerPro/1.0 (Streamlit)"}
_TIMEOUT = 25

# CoinGecko caps `per_page` at 250 (the original requested 1000, which the API
# silently clamps). We request the max so every tab has plenty of coins.
MAX_PER_PAGE = 250


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params or {}, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=120, show_spinner=False)
def get_markets(
    vs_currency: str = "inr",
    per_page: int = MAX_PER_PAGE,
    page: int = 1,
    sparkline: bool = True,
    order: str = "market_cap_desc",
) -> List[Dict[str, Any]]:
    """Market list (price, market cap, 24h change, 7d sparkline) for many coins.

    Equivalent to: /coins/markets?vs_currency=inr&order=market_cap_desc&...
    Cached for 2 minutes.
    """
    per_page = max(1, min(int(per_page), MAX_PER_PAGE))
    data = _get(
        "/coins/markets",
        {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": str(sparkline).lower(),
            "price_change_percentage": "24h,7d",
        },
    )
    return data if isinstance(data, list) else []


@st.cache_data(ttl=300, show_spinner=False)
def get_coin(coin_id: str) -> Optional[Dict[str, Any]]:
    """Full detail for a single coin (market data, community, developer, links).

    Equivalent to: /coins/{id}?localization=false&tickers=true&market_data=true...
    Cached for 5 minutes.
    """
    if not coin_id:
        return None
    data = _get(
        f"/coins/{coin_id}",
        {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "true",
        },
    )
    # Enforce the Optional[dict] contract so callers never .get() a non-dict.
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=900, show_spinner=False)
def get_market_chart(coin_id: str, vs_currency: str = "inr", days: int = 180) -> List[List[float]]:
    """Historical price series: list of ``[timestamp_ms, price]`` pairs.

    For ``days`` >= ~90 CoinGecko returns daily granularity, which is what the
    forecasting pipeline trains on. Cached for 15 minutes. Returns [] on failure.
    """
    if not coin_id:
        return []
    try:
        data = _get(
            f"/coins/{coin_id}/market_chart",
            {"vs_currency": vs_currency, "days": days},
        )
        prices = data.get("prices", []) if isinstance(data, dict) else []
        return [[float(ts), float(p)] for ts, p in prices if p is not None]
    except requests.RequestException:
        return []


def load_markets(**kwargs) -> List[Dict[str, Any]]:
    """Caller-friendly wrapper: shows a spinner and a clean error on failure."""
    try:
        with st.spinner("Loading live market data from CoinGecko…"):
            return get_markets(**kwargs)
    except requests.HTTPError as exc:  # pragma: no cover - network dependent
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 429:
            st.error("CoinGecko rate limit hit (HTTP 429). Please wait a minute and refresh.")
        else:
            st.error(f"Failed to load market data (HTTP {status}).")
        return []
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        st.error(f"Network error loading market data: {exc}")
        return []


def load_coin(coin_id: str) -> Optional[Dict[str, Any]]:
    """Caller-friendly wrapper for a single coin's full detail."""
    try:
        with st.spinner("Loading coin details…"):
            return get_coin(coin_id)
    except requests.HTTPError as exc:  # pragma: no cover - network dependent
        status = exc.response.status_code if exc.response is not None else "?"
        st.error(f"Failed to load coin details (HTTP {status}).")
        return None
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        st.error(f"Network error loading coin details: {exc}")
        return None
