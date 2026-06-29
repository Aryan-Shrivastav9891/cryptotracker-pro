"""Leakage-free technical features for the ML forecasters.

Every feature at row *i* uses ONLY information up to and including row *i*.
The supervised target is the NEXT-step log return, ``log(price[i+1]/price[i])``
(i.e. ``logret.shift(-1)``) — so a model trained on (features[i], target[i]) is
predicting the future, never peeking at it.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_features(prices: np.ndarray) -> pd.DataFrame:
    """Per-row technical features (causal — only past/current info)."""
    s = pd.Series(np.asarray(prices, dtype=float))
    ret = np.log(s / s.shift(1))
    out = pd.DataFrame(index=s.index)

    # Lagged returns (momentum).
    out["r1"] = ret
    out["r2"] = ret.shift(1)
    out["r3"] = ret.shift(2)
    out["r5"] = ret.shift(4)
    out["r10"] = ret.shift(9)

    # Moving-average gaps (trend).
    out["sma5_gap"] = s / s.rolling(5).mean() - 1.0
    out["sma10_gap"] = s / s.rolling(10).mean() - 1.0
    out["sma20_gap"] = s / s.rolling(20).mean() - 1.0
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    out["ema12_gap"] = s / ema12 - 1.0
    out["macd"] = (ema12 - ema26) / s

    # RSI(14) scaled to 0..1.
    delta = s.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    # np.maximum(al, eps) instead of replace(0, NaN): a pure uptrend (al==0) -> RSI 1.0,
    # not NaN — so trending rows aren't dropped (which would bias band calibration).
    out["rsi14"] = (100 - 100 / (1 + ag / np.maximum(al, 1e-10))) / 100.0

    # Bollinger z-score (mean reversion) + rolling volatility (risk).
    sma20, std20 = s.rolling(20).mean(), s.rolling(20).std()
    out["boll_z"] = (s - sma20) / std20.replace(0, np.nan)
    out["vol10"] = ret.rolling(10).std()
    out["vol20"] = ret.rolling(20).std()

    # MA slopes (trend strength) + a trend-vs-range regime flag — all causal.
    out["sma10_slope"] = s.rolling(10).mean().diff() / s
    out["sma20_slope"] = s.rolling(20).mean().diff() / s
    out["regime"] = (out["sma20_slope"].abs() / out["vol20"].replace(0, np.nan)).clip(0, 5)
    return out


def build_xy(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix X and next-step log-return target y (NaN rows dropped)."""
    s = pd.Series(np.asarray(prices, dtype=float))
    feats = compute_features(prices)
    target = np.log(s.shift(-1) / s)  # next-step log return — no leakage
    df = feats.copy()
    df["__y"] = target.values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return np.empty((0, feats.shape[1])), np.empty((0,))
    return df.drop(columns="__y").values, df["__y"].values


def latest_row(prices: np.ndarray) -> np.ndarray:
    """Feature vector for the most recent price (for iterative multi-step prediction)."""
    return compute_features(prices).iloc[-1].to_numpy(dtype=float)
