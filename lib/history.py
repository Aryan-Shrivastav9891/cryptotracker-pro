"""Daily price history for forecasting.

Primary source: **Binance daily OHLCV via ccxt** (key-less, ~2 years available).
Coin id/symbol is mapped to a ``<SYM>/USDT`` pair. If ccxt is unavailable, the
pair doesn't exist, or Binance is geo-blocked, it falls back to CoinGecko's
``market_chart`` (which caps at ~365 days). Cached for 1 hour.
"""
from __future__ import annotations

import time
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

    return _coingecko_daily_fallback(coin_id, days)


def _coingecko_daily_fallback(coin_id: str, days: int) -> Tuple[List[datetime], List[float], str]:
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


# --------------------------------------------------------------------------- #
# Intraday (hourly) — for the Intraday Signal Lab
# --------------------------------------------------------------------------- #
def _ccxt_hourly(symbol: str, hours: int) -> Optional[Tuple[List[datetime], np.ndarray]]:
    ex = _exchange()
    if ex is None:
        return None
    pair = f"{symbol.upper()}/USDT"
    ex.load_markets()
    if pair not in ex.markets:
        return None
    now = ex.milliseconds()
    cur = now - int((hours + 2) * 3600 * 1000)
    rows: List[list] = []
    for _ in range(40):  # pagination guard (40 * 1000 candles is plenty)
        batch = ex.fetch_ohlcv(pair, timeframe="1h", since=cur, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 3600 * 1000
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1000:
            break
        time.sleep(max(ex.rateLimit / 1000.0, 0.1))  # rate-limit-safe
    if not rows:
        return None
    seen, dedup = set(), []
    for c in rows:
        if c[0] not in seen:
            seen.add(c[0])
            dedup.append(c)
    dedup.sort(key=lambda c: c[0])
    dates = [datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc) for c in dedup]
    closes = np.asarray([c[4] for c in dedup], dtype=float)
    return dates, closes


@st.cache_data(ttl=1800, show_spinner=False)
def get_hourly_closes(
    coin_id: str, symbol: str, hours: int = 720
) -> Tuple[List[datetime], List[float], str]:
    """Hourly closes (Binance via ccxt, else CoinGecko hourly). Cached 30 min."""
    if HAS_CCXT and symbol:
        try:
            res = _ccxt_hourly(symbol, hours)
            if res is not None and len(res[1]) >= 120:
                dates, closes = res
                return dates[-hours:], [float(x) for x in closes[-hours:]], "Binance (ccxt)"
        except Exception:
            pass
    # CoinGecko returns hourly granularity for 2–90 day windows.
    days = max(2, min(90, int(np.ceil(hours / 24)) + 1))
    raw = coingecko.get_market_chart(coin_id, days=days)
    if raw:
        dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in raw][-hours:]
        closes = [float(p) for _, p in raw][-hours:]
        return dates, closes, "CoinGecko (hourly)"
    return [], [], "none"
