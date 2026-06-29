"""Fast per-coin price estimate for the card grids (home / altcoins / predicted…).

The original pooled every coin into one model with a single global min-max
normalisation — which collapses small-cap coins to ~0 and makes the "prediction"
meaningless. This replaces it with a per-coin, scale-correct estimate: fit a
log-linear trend to each coin's recent price action and project ~1 day ahead,
capping the move to a sane range.

For a proper, backtested multi-day forecast with confidence intervals, see
``lib/forecast.py`` (used by the Future page).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

try:  # only used to describe deep-forecast availability in the UI
    import statsmodels  # noqa: F401

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

WINDOW = 24            # recent hourly sparkline points used to estimate the trend
HORIZON = 24           # project ~1 day ahead (sparkline is hourly over 7 days)
MAX_STEP_CHANGE = 0.5  # cap projected move at ±50% to avoid absurd extrapolation
ENGINE = "Per-coin log-trend projection (≈1-day)"


def _sparkline(coin: dict) -> List[float]:
    spark = (coin.get("sparkline_in_7d") or {}).get("price") or []
    return [float(p) for p in spark if p is not None]


def _signature(coins: List[dict]) -> Tuple:
    """Stable, hashable representation of the inputs for caching."""
    return tuple(
        (coin.get("id"), float(coin.get("current_price") or 0.0), tuple(_sparkline(coin)))
        for coin in coins
    )


def _trend_predict(signature: Tuple) -> Dict[str, float]:
    preds: Dict[str, float] = {}
    for coin_id, current_price, prices in signature:
        series = [p for p in prices if p > 0]
        if len(series) >= 8:
            window = series[-WINDOW:] if len(series) > WINDOW else series
            n = len(window)
            t = np.arange(n, dtype=float)
            slope, intercept = np.polyfit(t, np.log(window), 1)
            last = window[-1]
            projected = float(np.exp(slope * (n - 1 + HORIZON) + intercept))
            change = (projected - last) / last if last else 0.0
            change = max(-MAX_STEP_CHANGE, min(MAX_STEP_CHANGE, change))
            value = last * (1 + change)
            preds[coin_id] = float(value) if np.isfinite(value) else float(current_price)
        else:
            preds[coin_id] = float(current_price)
    return preds


@st.cache_data(ttl=300, show_spinner=False)
def _predict_cached(signature: Tuple) -> Dict[str, float]:
    return _trend_predict(signature)


def predict_prices(
    coins: List[dict], window: int = WINDOW, prefer_deep: bool = True
) -> Tuple[Dict[str, float], str]:
    """Estimate each coin's price ~1 day ahead.

    Returns ``(predictions_by_id, engine_name)``. ``window``/``prefer_deep`` are
    accepted for backward compatibility.
    """
    if not coins:
        return {}, ENGINE
    return _predict_cached(_signature(coins)), ENGINE


def model_status() -> str:
    """Human-readable note for the sidebar about the prediction engines."""
    deep = (
        "statsmodels available — deep, backtested forecasts on the Future page."
        if HAS_STATSMODELS
        else "Install statsmodels for deep forecasts on the Future page."
    )
    return f"📈 Card estimates: {ENGINE}. {deep}"
