"""Attach derived fields (AI prediction, future profitability) to coin dicts,
plus shared filtering helpers used across the market views.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lib import model
from lib.formatting import safe_float


def _sparkline(coin: Dict[str, Any]) -> List[float]:
    spark = (coin.get("sparkline_in_7d") or {}).get("price") or []
    return [float(p) for p in spark if p is not None]


def enrich_coins(coins: List[Dict[str, Any]], prefer_deep: bool = True) -> Tuple[List[Dict[str, Any]], str]:
    """Return shallow copies of ``coins`` with prediction fields added.

    Adds: predicted_price, predicted_change (%), future_profitability (%).
    Copies each dict first so the cached CoinGecko response is never mutated.
    """
    preds, engine = model.predict_prices(coins, prefer_deep=prefer_deep)
    enriched: List[Dict[str, Any]] = []
    for coin in coins:
        c = dict(coin)
        current = safe_float(c.get("current_price"))
        pred = preds.get(c.get("id"))
        c["predicted_price"] = pred
        # `pred is not None` so a legitimately-predicted 0.0 still yields -100%.
        c["predicted_change"] = (
            ((pred - current) / current * 100.0) if (pred is not None and current) else 0.0
        )

        spark = _sparkline(c)
        if len(spark) > 1 and spark[0]:
            c["future_profitability"] = (spark[-1] - spark[0]) / spark[0] * 100.0
        else:
            c["future_profitability"] = 0.0

        # Human-readable buy/hold note shown on coin cards (ports index.js).
        fp = c["future_profitability"]
        c["description"] = (
            f"This coin has shown strong growth potential with a projected "
            f"profitability of {fp:.2f}%. It is a great option for short-term gains."
            if fp > 20
            else "This coin has steady performance and is a safer option for gradual profits."
        )
        enriched.append(c)
    return enriched, engine


def filter_by_search(coins: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = (query or "").lower().strip()
    if not q:
        return coins
    return [
        c
        for c in coins
        if q in (c.get("name") or "").lower() or q in (c.get("symbol") or "").lower()
    ]
