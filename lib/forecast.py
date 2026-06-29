"""Real-world price forecasting pipeline (single coin, done properly).

This replaces the original toy approach (pool every coin, normalise globally,
"predict" the price you already trained on). The flow here is how short-horizon
forecasting is actually done:

  1. Pull REAL historical daily prices from CoinGecko (market_chart).
  2. BACKTEST with a walk-forward holdout to measure honest accuracy on data the
     model never saw — MAE, MAPE and directional (up/down) accuracy.
  3. Refit on the full history and forecast `horizon` days ahead, with a
     confidence band derived from the backtest errors.

Primary model: Holt's damped-trend Exponential Smoothing (statsmodels) — a
robust, fast, interpretable workhorse for trended series that returns smooth
multi-step forecasts. Falls back to a log-return drift model if statsmodels is
unavailable or a fit fails, so a forecast is always produced.

Nothing here can truly "beat the market" — crypto is largely a random walk. The
value is an honest, backtested estimate WITH its measured error, not a magic
number. Not financial advice.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from lib import coingecko

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    ExponentialSmoothing = None  # type: ignore
    HAS_STATSMODELS = False

Z_80 = 1.2816        # ~80% prediction interval
MAX_PCT_ERROR = 5.0  # cap a single backtest point's % error (500%) so a tiny
                     # price denominator can't blow MAPE up into the thousands


# --------------------------------------------------------------------------- #
# Base forecasters
# --------------------------------------------------------------------------- #
def _drift_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Geometric (log-return) drift — the honest random-walk-with-drift baseline."""
    series = np.asarray(series, dtype=float)
    pos = series[series > 0]
    if len(pos) < 2:
        last = float(series[-1]) if len(series) else 0.0
        return np.full(horizon, last)
    log_returns = np.diff(np.log(pos))
    mu = float(np.mean(log_returns))
    last = float(pos[-1])
    steps = np.arange(1, horizon + 1)
    return last * np.exp(mu * steps)


def _holt_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Holt damped-trend exponential smoothing; raises on failure."""
    series = np.asarray(series, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series, trend="add", damped_trend=True, initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        fc = np.asarray(fit.forecast(horizon), dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite forecast")
    return np.clip(fc, 0.0, None)


def _forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Best available forecaster with graceful fallback."""
    series = np.asarray(series, dtype=float)
    if HAS_STATSMODELS and len(series) >= 12:
        try:
            return _holt_forecast(series, horizon)
        except Exception:
            pass
    return _drift_forecast(series, horizon)


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def _backtest(series: np.ndarray, horizon: int, folds: int = 4) -> Dict[str, Any]:
    """Evaluate the forecaster on held-out windows it never trained on."""
    n = len(series)
    min_train = 20
    step_errors: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}
    abs_errors: List[float] = []
    pct_errors: List[float] = []
    dir_correct = 0
    dir_total = 0
    used_folds = 0

    for k in range(folds, 0, -1):
        origin = n - horizon * k
        if origin < min_train:
            continue
        train = series[:origin]
        actual = series[origin : origin + horizon]
        if len(actual) < horizon:
            continue
        try:
            pred = _forecast(train, horizon)
        except Exception:
            continue
        used_folds += 1
        for h in range(horizon):
            err = float(actual[h] - pred[h])
            step_errors[h + 1].append(err)
            abs_errors.append(abs(err))
            if actual[h] != 0:
                # Cap the per-point % error so one tiny denominator can't dominate.
                pct_errors.append(min(abs(err) / abs(actual[h]), MAX_PCT_ERROR))
        # Directional accuracy: did we get the end-of-horizon direction right?
        # Skip flat folds (no actual move) — there's no direction to predict, and
        # counting sign(0)==sign(0) as "correct" would inflate the metric.
        if actual[-1] != train[-1]:
            pred_dir = np.sign(pred[-1] - train[-1])
            actual_dir = np.sign(actual[-1] - train[-1])
            dir_total += 1
            if pred_dir == actual_dir:
                dir_correct += 1

    mae = float(np.mean(abs_errors)) if abs_errors else None
    mape = float(np.mean(pct_errors)) if pct_errors else None
    directional = (dir_correct / dir_total) if dir_total else None

    # Per-step sigma for the confidence band; fall back to scaled base sigma.
    base_sigma = float(np.std(abs_errors)) if len(abs_errors) >= 2 else 0.0
    sigma_steps: List[float] = []
    for h in range(1, horizon + 1):
        errs = step_errors[h]
        if len(errs) >= 2:
            sigma_steps.append(float(np.std(errs)))
        else:
            sigma_steps.append(base_sigma * np.sqrt(h))

    return {
        "mae": mae,
        "mape": mape,
        "directional": directional,
        "sigma_steps": sigma_steps,
        "folds": used_folds,
    }


def _confidence(mape: Optional[float], directional: Optional[float]) -> float:
    """0-100 confidence from measured error — deliberately conservative.

    Crypto is close to a random walk over short horizons, so this is tuned NOT to
    overpromise: 50% directional accuracy (a coin flip) scores zero directional
    skill, and the price-error term reaches zero at 20% MAPE.
    """
    if mape is None and directional is None:
        return 0.0
    price_score = max(0.0, 1.0 - (mape if mape is not None else 0.5) / 0.20)  # 20% MAPE -> 0
    # Reward directional accuracy only ABOVE chance (0.5); a coin flip -> 0.
    dir_score = max(0.0, (directional - 0.5) / 0.5) if directional is not None else 0.0
    return float(max(0.0, min(100.0, 100.0 * (0.5 * price_score + 0.5 * dir_score))))


def recommendation(pct_change: float, confidence: float) -> str:
    if pct_change > 10 and confidence > 60:
        return "Strong Buy"
    if pct_change > 4 and confidence > 55:
        return "Buy"
    if pct_change > -4:
        return "Hold"
    if pct_change > -12:
        return "Watch"
    return "Sell"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner=False)
def forecast_coin(coin_id: str, days: int = 180, horizon: int = 7) -> Optional[Dict[str, Any]]:
    """Backtest + forecast one coin. Returns a JSON-friendly result dict, or None.

    Returns ``None`` when there is too little history to run even one walk-forward
    backtest fold (roughly < ``20 + horizon`` daily points) — we never present an
    unvalidated forecast.

    Result keys: engine, days, horizon, history_dates, history_prices,
    forecast_dates, forecast_prices, lower, upper, current_price,
    predicted_price, predicted_change, mae, mape, directional, confidence,
    recommendation, folds, band_clipped.
    """
    raw = coingecko.get_market_chart(coin_id, days=days)
    if len(raw) < 15:
        return None

    timestamps = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in raw]
    prices = np.asarray([p for _, p in raw], dtype=float)

    bt = _backtest(prices, horizon)
    if bt["folds"] == 0:
        # No walk-forward fold was possible (too little history for this horizon),
        # so we can neither validate the model nor size its uncertainty band.
        # Refuse rather than present an unvalidated, falsely-precise forecast.
        return None

    forecast = _forecast(prices, horizon)

    sigma = bt["sigma_steps"]
    raw_lower = [float(forecast[h] - Z_80 * sigma[h]) for h in range(horizon)]
    lower = [max(0.0, x) for x in raw_lower]
    upper = [float(forecast[h] + Z_80 * sigma[h]) for h in range(horizon)]
    # Flag when the band was clamped at the ₹0 price floor: the interval is then
    # asymmetric and its true coverage is below the nominal 80%.
    band_clipped = any(x < 0 for x in raw_lower)

    last_date = timestamps[-1]
    forecast_dates = [last_date + timedelta(days=h + 1) for h in range(horizon)]

    current_price = float(prices[-1])
    predicted_price = float(forecast[-1])
    predicted_change = (
        (predicted_price - current_price) / current_price * 100.0 if current_price else 0.0
    )
    confidence = _confidence(bt["mape"], bt["directional"])

    engine = (
        "Holt damped-trend Exponential Smoothing (statsmodels)"
        if HAS_STATSMODELS
        else "Log-return drift (random-walk baseline)"
    )

    return {
        "engine": engine,
        "days": days,
        "horizon": horizon,
        "history_dates": timestamps,
        "history_prices": [float(p) for p in prices],
        "forecast_dates": forecast_dates,
        "forecast_prices": [float(f) for f in forecast],
        "lower": lower,
        "upper": upper,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_change": predicted_change,
        "mae": bt["mae"],
        "mape": bt["mape"],
        "directional": bt["directional"],
        "confidence": confidence,
        "recommendation": recommendation(predicted_change, confidence),
        "folds": bt["folds"],
        "band_clipped": band_clipped,
    }
