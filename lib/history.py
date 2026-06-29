"""Daily price history for forecasting.

Primary source: **Binance daily OHLCV via ccxt** (key-less, ~2 years available).
Coin id/symbol is mapped to a ``<SYM>/USDT`` pair. If ccxt is unavailable, the
pair doesn't exist, or Binance is geo-blocked, it falls back to CoinGecko's
``market_chart`` (which caps at ~365 days). Cached for 1 hour.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

from lib import coingecko

try:
    import ccxt  # type: ignore

    HAS_CCXT = True
except Exception:  # pragma: no cover
    ccxt = None  # type: ignore
    HAS_CCXT = False

_EXCHANGE = None  # lazily-built singleton (reuses ccxt's per-instance market cache)


def _exchange():
    global _EXCHANGE
    if _EXCHANGE is None and HAS_CCXT:
        _EXCHANGE = ccxt.binance({"enableRateLimit": True, "timeout": 30000})
    return _EXCHANGE


def _ccxt_daily(symbol: str, days: int) -> Optional[Tuple[List[datetime], np.ndarray]]:
    ex = _exchange()
    if ex is None:
        return None
    pair = f"{symbol.upper()}/USDT"
    ex.load_markets()
    if pair not in ex.markets:
        return None
    since = ex.milliseconds() - int((days + 5) * 86400 * 1000)
    ohlcv = ex.fetch_ohlcv(pair, timeframe="1d", since=since, limit=1000)
    if not ohlcv:
        return None
    dates = [datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc) for c in ohlcv]
    closes = np.asarray([c[4] for c in ohlcv], dtype=float)
    return dates, closes


@st.cache_data(ttl=3600, show_spinner=False)
def get_daily_closes(
    coin_id: str, symbol: str, days: int = 540
) -> Tuple[List[datetime], List[float], str]:
    """Return ``(dates, closes, source)`` of daily closing prices.

    ``source`` is "Binance (ccxt)", "CoinGecko" or "none". Never raises.
    """
    # 1) Try Binance via ccxt (more history, free, no key).
    if HAS_CCXT and symbol:
        try:
            res = _ccxt_daily(symbol, days)
            if res is not None and len(res[1]) >= 60:
                dates, closes = res
                return dates, [float(x) for x in closes], "Binance (ccxt)"
        except Exception:
            pass  # geo-block / network / unknown pair -> fall back

    # 2) Fall back to CoinGecko market_chart (caps ~365 days).
    capped = min(int(days), 365)
    raw = coingecko.get_market_chart(coin_id, days=capped)
    if raw:
        dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in raw]
        closes = [float(p) for _, p in raw]
        # Be transparent when the requested window was clamped to CoinGecko's limit.
        label = "CoinGecko (365d max)" if int(days) > 365 else "CoinGecko"
        return dates, closes, label

    return [], [], "none"
