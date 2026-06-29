#!/usr/bin/env python3
"""Download multi-year OHLCV crypto data from Binance and engineer ML features.

Uses ccxt's PUBLIC market-data endpoints, which need **no API key** and are free.
Handles pagination, rate limiting, retries, de-duplication, and adds common
technical-indicator features, then writes one CSV per symbol.

Install:
    pip install ccxt pandas numpy

Run:
    python scripts/fetch_crypto_data.py

Note on geo-blocking: `binance.com` is unavailable from some regions (HTTP 451).
If so, set EXCHANGE_ID to a reachable alternative that also needs no key, e.g.
"binanceus", "kraken", or "coinbase" (symbols may differ, e.g. BTC/USD on Kraken).
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import ccxt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — edit these
# --------------------------------------------------------------------------- #
EXCHANGE_ID = "binance"                 # any ccxt exchange id with public OHLCV
SYMBOLS: List[str] = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1d"                        # "1d", "4h", "1h", "15m", ...
START_DATE = "2022-01-01"              # UTC; ~2-3 years back is a good default
OUTPUT_DIR = "data"                     # one CSV per symbol is written here

CANDLES_PER_CALL = 1000                 # Binance max per request (1000)
MAX_RETRIES = 5                         # retries per failed request
RETRY_BACKOFF = 2.0                     # seconds, doubled each retry
DROP_INCOMPLETE_LAST = True            # drop the still-forming final candle
DROP_WARMUP_NANS = False               # drop early rows where indicators are NaN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch")


# --------------------------------------------------------------------------- #
# Exchange setup
# --------------------------------------------------------------------------- #
def build_exchange() -> ccxt.Exchange:
    """Create a ccxt exchange client with built-in rate limiting enabled."""
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class(
        {
            "enableRateLimit": True,  # ccxt throttles automatically...
            "timeout": 30000,         # ...and we also sleep manually below.
        }
    )
    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError(f"{EXCHANGE_ID} does not support fetchOHLCV")
    return exchange


def with_retries(fn, *args, **kwargs):
    """Call ``fn`` with retries + exponential backoff on transient API errors."""
    delay = RETRY_BACKOFF
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout, ccxt.DDoSProtection,
                ccxt.RateLimitExceeded) as err:
            last_err = err
            log.warning("  request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, MAX_RETRIES, type(err).__name__, delay)
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except ccxt.ExchangeError as err:
            # Non-transient (bad symbol, etc.) — don't hammer the API.
            raise
    raise RuntimeError(f"Giving up after {MAX_RETRIES} retries") from last_err


# --------------------------------------------------------------------------- #
# Paginated OHLCV download
# --------------------------------------------------------------------------- #
def fetch_ohlcv_paginated(exchange: ccxt.Exchange, symbol: str,
                          timeframe: str, since_ms: int) -> pd.DataFrame:
    """Page through OHLCV from ``since_ms`` to now.

    ccxt returns at most ~1000 candles per call, so we advance ``since`` to just
    after the last candle each iteration and loop until we reach the present.
    """
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000  # seconds -> ms
    now_ms = exchange.milliseconds()
    all_rows: List[list] = []
    since = since_ms

    while since < now_ms:
        batch = with_retries(
            exchange.fetch_ohlcv, symbol, timeframe, since, CANDLES_PER_CALL
        )
        if not batch:
            break  # no more data

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        # Advance past the last candle to avoid re-fetching it next loop.
        next_since = last_ts + timeframe_ms
        if next_since <= since:
            break  # safety: no forward progress -> stop

        since = next_since
        log.info("  %s: %d candles so far (through %s)",
                 symbol, len(all_rows),
                 pd.to_datetime(last_ts, unit="ms").strftime("%Y-%m-%d %H:%M"))

        # Respect the exchange rate limit (in addition to ccxt's own throttling).
        time.sleep(max(exchange.rateLimit / 1000.0, 0.2))

        # If the exchange returned fewer than a full page, we're at the end.
        if len(batch) < CANDLES_PER_CALL:
            break

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_rows, columns=cols)
    return df


def tidy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to a clean, datetime-indexed, de-duplicated, sorted frame."""
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"])
    df = df[~df.index.duplicated(keep="first")]  # drop duplicate timestamps
    df = df.sort_index()                          # ensure chronological order
    # Numeric, just in case the API returned strings.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Feature engineering (technical indicators)
# --------------------------------------------------------------------------- #
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (ewm, alpha=1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common predictive features: returns, MAs, RSI, MACD, BB, volatility."""
    if df.empty:
        return df
    out = df.copy()
    close = out["close"]

    # --- Returns ---
    out["return_1"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))

    # --- Moving averages (trend) ---
    for w in (7, 21, 50):
        out[f"sma_{w}"] = close.rolling(w).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- MACD (trend/momentum) ---
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # --- RSI (momentum) ---
    out["rsi_14"] = compute_rsi(close, 14)

    # --- Bollinger Bands (mean reversion) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_mid_20"] = bb_mid
    out["bb_upper_20"] = bb_mid + 2 * bb_std
    out["bb_lower_20"] = bb_mid - 2 * bb_std
    # Position within the bands in [0, 1] (handy normalized feature).
    out["bb_pct_20"] = (close - out["bb_lower_20"]) / (
        (out["bb_upper_20"] - out["bb_lower_20"]).replace(0, np.nan)
    )

    # --- Rolling volatility (risk) ---
    for w in (7, 30):
        out[f"volatility_{w}"] = out["log_return"].rolling(w).std()

    # --- Volume context ---
    out["volume_sma_20"] = out["volume"].rolling(20).mean()

    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exchange = build_exchange()
    with_retries(exchange.load_markets)  # validate connectivity + symbols
    since_ms = exchange.parse8601(f"{START_DATE}T00:00:00Z")

    for symbol in SYMBOLS:
        log.info("Downloading %s %s from %s on %s…",
                 symbol, TIMEFRAME, START_DATE, EXCHANGE_ID)
        if symbol not in exchange.markets:
            log.error("  symbol %s not available on %s — skipping", symbol, EXCHANGE_ID)
            continue

        raw = fetch_ohlcv_paginated(exchange, symbol, TIMEFRAME, since_ms)
        df = tidy_frame(raw)
        if df.empty:
            log.error("  no data returned for %s — skipping", symbol)
            continue

        # Drop the final, still-forming candle (its values keep changing).
        if DROP_INCOMPLETE_LAST and len(df) > 1:
            df = df.iloc[:-1]

        df = add_features(df)
        if DROP_WARMUP_NANS:
            df = df.dropna()

        out_path = os.path.join(OUTPUT_DIR, f"{safe_filename(symbol)}_{TIMEFRAME}.csv")
        df.to_csv(out_path, index_label="datetime")
        log.info("  ✓ saved %d rows × %d cols -> %s  (%s → %s)",
                 len(df), df.shape[1], out_path,
                 df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))

    log.info("Done.")


if __name__ == "__main__":
    main()
