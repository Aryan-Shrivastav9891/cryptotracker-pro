"""Ensemble price forecasting with honest, out-of-sample backtesting.

The goal is NOT a flashy accuracy number — daily crypto prices are close to a
random walk. The goal is a forecast that *provably beats a naive baseline on data
it never saw*, with the UI told the truth when it doesn't.

Pipeline (per coin):
  1. Pull 1.5–2 years of daily closes (lib.history: Binance via ccxt, else CoinGecko).
  2. Build an ENSEMBLE of models, each `fn(train, h) -> price forecast`:
       • naive   — random-walk-with-drift (the baseline everything must beat)
       • holt    — Holt damped-trend Exponential Smoothing on log price
       • theta   — ThetaModel
       • arima   — ARIMA on log price (order picked once by AIC)
       • gbr     — HistGradientBoosting on engineered features → next log return
     Models combine with weights = (1 / backtest-MAPE), normalised.
  3. BACKTEST: expanding-window walk-forward, one-step-ahead, refit each step,
     over the last ~60 days. Report MAPE / MAE / RMSE / directional accuracy per
     model AND for the ensemble, plus SKILL = (naive_MAPE − ens_MAPE)/naive_MAPE.
  4. Reliability: skill > 0 AND directional > 50%. Otherwise the signal says so.
  5. Forecast `horizon` days ahead; 80% band via Monte-Carlo bootstrap of the
     ensemble's one-step relative errors, compounded over the horizon.

Optional deps degrade gracefully: without statsmodels/sklearn the ensemble simply
uses fewer models (always at least the naive baseline).
Not financial advice.
"""
from __future__ import annotations

import warnings
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from lib import history

# ---- optional libraries -----------------------------------------------------
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
    from statsmodels.tsa.arima.model import ARIMA

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

LOOKBACK = 60          # walk-forward one-step backtest length (days)
MIN_TRAIN = 60         # smallest training window (also GBR's minimum)
MAX_PCT_ERROR = 5.0    # cap a single point's % error so tiny prices can't blow up MAPE
RET_CLIP = 0.25        # clamp per-step predicted log-return to ±25%
MC_SIMS = 2000         # Monte-Carlo simulations for the confidence band


# =========================================================================== #
# Base forecasters: fn(train: np.ndarray, h: int) -> np.ndarray of h prices
# =========================================================================== #
def _naive(train: np.ndarray, h: int) -> np.ndarray:
    """Random-walk-with-drift (geometric)."""
    s = np.asarray(train, dtype=float)
    pos = s[s > 0]
    if len(pos) < 2:
        last = float(s[-1]) if len(s) else 0.0
        return np.full(h, last)
    mu = float(np.mean(np.diff(np.log(pos))))
    last = float(pos[-1])
    return last * np.exp(mu * np.arange(1, h + 1))


def _holt(train: np.ndarray, h: int) -> np.ndarray:
    s = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            np.log(s), trend="add", damped_trend=True, initialization_method="estimated"
        ).fit(optimized=True)
        fc = np.exp(np.asarray(fit.forecast(h), dtype=float))
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite holt forecast")
    return np.clip(fc, 0.0, None)


def _theta(train: np.ndarray, h: int) -> np.ndarray:
    s = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ThetaModel(s, period=7, deseasonalize=False).fit()
        fc = np.asarray(fit.forecast(h), dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite theta forecast")
    return np.clip(fc, 0.0, None)


def _make_arima(order: Tuple[int, int, int]) -> Callable[[np.ndarray, int], np.ndarray]:
    def _arima(train: np.ndarray, h: int) -> np.ndarray:
        s = np.asarray(train, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(np.log(s), order=order).fit()
            fc = np.exp(np.asarray(fit.forecast(h), dtype=float))
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite arima forecast")
        return np.clip(fc, 0.0, None)

    return _arima


def _best_arima_order(train: np.ndarray) -> Tuple[int, int, int]:
    arr = np.asarray(train, dtype=float)
    arr = arr[arr > 0]  # avoid log(0) -> -inf RuntimeWarning
    if len(arr) < 10:
        return (1, 1, 1)
    logp = np.log(arr)
    best, best_aic = (1, 1, 1), np.inf
    for order in [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aic = ARIMA(logp, order=order).fit().aic
            if np.isfinite(aic) and aic < best_aic:
                best, best_aic = order, aic
        except Exception:
            continue
    return best


# ---- GBR on engineered features --------------------------------------------
def _compute_features(prices: np.ndarray) -> pd.DataFrame:
    """Technical features per row, using ONLY information up to that row."""
    s = pd.Series(np.asarray(prices, dtype=float))
    ret = np.log(s / s.shift(1))
    out = pd.DataFrame(index=s.index)
    out["r1"], out["r2"], out["r3"], out["r5"] = ret, ret.shift(1), ret.shift(2), ret.shift(4)
    out["sma5_gap"] = s / s.rolling(5).mean() - 1.0
    out["sma10_gap"] = s / s.rolling(10).mean() - 1.0
    out["sma20_gap"] = s / s.rolling(20).mean() - 1.0
    ema12, ema26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    out["ema12_gap"] = s / ema12 - 1.0
    out["macd"] = (ema12 - ema26) / s
    delta = s.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["rsi14"] = (100 - 100 / (1 + ag / al.replace(0, np.nan))) / 100.0
    sma20, std20 = s.rolling(20).mean(), s.rolling(20).std()
    out["boll_z"] = (s - sma20) / std20.replace(0, np.nan)
    out["vol10"] = ret.rolling(10).std()
    return out


def _gbr(train: np.ndarray, h: int) -> np.ndarray:
    if not HAS_SKLEARN or len(train) < MIN_TRAIN:
        raise RuntimeError("gbr unavailable")
    s = pd.Series(np.asarray(train, dtype=float))
    feats = _compute_features(train)
    target = np.log(s.shift(-1) / s)  # next-step log return (no leakage)
    df = feats.copy()
    df["__y"] = target.values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 30:
        raise RuntimeError("gbr: not enough rows")
    X, y = df.drop(columns="__y").values, df["__y"].values
    model = HistGradientBoostingRegressor(
        max_depth=3, max_iter=150, learning_rate=0.06,
        l2_regularization=1.0, min_samples_leaf=20, random_state=42,
    )
    model.fit(X, y)
    # Iterative multi-step: predict next return, append, recompute features.
    series = list(map(float, train))
    out: List[float] = []
    for _ in range(h):
        row = _compute_features(np.asarray(series)).iloc[-1].values.reshape(1, -1)
        ret_hat = 0.0 if np.isnan(row).any() else float(model.predict(row)[0])
        ret_hat = float(np.clip(ret_hat, -RET_CLIP, RET_CLIP))
        nxt = series[-1] * np.exp(ret_hat)
        out.append(nxt)
        series.append(nxt)
    return np.asarray(out, dtype=float)


# =========================================================================== #
# Backtest + metrics
# =========================================================================== #
def _metrics(preds: np.ndarray, actuals: np.ndarray, prevs: np.ndarray) -> Dict[str, Optional[float]]:
    """Metrics over the steps where the model actually produced a prediction.

    NaN predictions (a model that failed that step) are ignored — they no longer
    masquerade as a perfect carry-forward.
    """
    preds, actuals, prevs = (np.asarray(a, dtype=float) for a in (preds, actuals, prevs))
    mask = np.isfinite(preds) & np.isfinite(actuals)
    if not mask.any():
        return {"mape": None, "mae": None, "rmse": None, "directional": None, "coverage": 0.0}
    p, a, pv = preds[mask], actuals[mask], prevs[mask]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = a != 0
    mape = float(np.mean(np.minimum(np.abs(err[nz]) / np.abs(a[nz]), MAX_PCT_ERROR))) if nz.any() else None
    moved = a != pv  # skip flat days (no direction to call)
    directional = (
        float(np.mean(np.sign(p[moved] - pv[moved]) == np.sign(a[moved] - pv[moved])))
        if moved.any() else None
    )
    return {"mape": mape, "mae": mae, "rmse": rmse, "directional": directional,
            "coverage": float(mask.mean())}


def _build_models(initial_train: np.ndarray) -> List[Tuple[str, Callable[[np.ndarray, int], np.ndarray]]]:
    """Candidate models that actually work on this data (validated on initial train)."""
    candidates: List[Tuple[str, Callable]] = [("naive", _naive)]
    if HAS_STATSMODELS:
        candidates.append(("holt", _holt))
        candidates.append(("theta", _theta))
        candidates.append(("arima", _make_arima(_best_arima_order(initial_train))))
    if HAS_SKLEARN:
        candidates.append(("gbr", _gbr))
    working: List[Tuple[str, Callable]] = []
    for name, fn in candidates:
        try:
            out = fn(initial_train, 1)
            if np.all(np.isfinite(out)) and out[0] > 0:
                working.append((name, fn))
        except Exception:
            continue
    return working


def _walk_forward(prices: np.ndarray, models, lookback: int):
    """Expanding-window, one-step-ahead, refit each step over the last `lookback` days."""
    n = len(prices)
    start = n - lookback
    preds: Dict[str, List[float]] = {name: [] for name, _ in models}
    actuals, prevs = [], []
    for t in range(start, n):
        train, actual, prev = prices[:t], float(prices[t]), float(prices[t - 1])
        actuals.append(actual)
        prevs.append(prev)
        for name, fn in models:
            try:
                p = float(fn(train, 1)[0])
            except Exception:
                p = np.nan  # NaN (not carry-forward) so a failed step isn't scored as perfect
            if not np.isfinite(p) or p <= 0:
                p = np.nan
            preds[name].append(p)
    return preds, np.asarray(actuals), np.asarray(prevs)


# =========================================================================== #
# Signal
# =========================================================================== #
_SIG_ORDER = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def make_signal(expected_return: float, mape: Optional[float], reliable: bool) -> Tuple[str, str]:
    if not reliable:
        return "Hold", ("No measured edge over a naive baseline (skill ≤ 0 or directional ≤ 50% "
                        "out-of-sample) — Hold.")
    thr = mape if (mape and mape > 0) else 0.05
    er = expected_return
    if er >= 2 * thr:
        sig = "Strong Buy"
    elif er >= thr:
        sig = "Buy"
    elif er <= -2 * thr:
        sig = "Strong Sell"
    elif er <= -thr:
        sig = "Sell"
    else:
        return "Hold", (f"Expected move {er:+.1%} is within the model's typical error "
                        f"(±{thr:.1%}) — Hold.")
    return sig, (f"Expected {er:+.1%} over the horizon vs typical one-step error ±{thr:.1%}.")


def tilt_signal(signal: str, sentiment_label: str) -> str:
    """Soft ±1-notch nudge from news sentiment (never the main driver)."""
    if signal not in _SIG_ORDER:
        return signal
    i = _SIG_ORDER.index(signal)
    if sentiment_label == "Bullish":
        i = min(i + 1, len(_SIG_ORDER) - 1)
    elif sentiment_label == "Bearish":
        i = max(i - 1, 0)
    return _SIG_ORDER[i]


# =========================================================================== #
# Public entry point
# =========================================================================== #
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_coin(
    coin_id: str, symbol: str, days: int = 540, horizon: int = 7
) -> Optional[Dict[str, Any]]:
    """Backtest + ensemble forecast for one coin. Returns a result dict, or None
    when there isn't enough history (~80 daily points) to backtest honestly.
    """
    horizon = int(max(1, min(horizon, 30)))
    dates, closes, source = history.get_daily_closes(coin_id, symbol, days=days)
    prices = np.asarray(closes, dtype=float)

    # Drop any non-finite / non-positive points (rare bad data) and keep dates aligned.
    if len(prices):
        good = np.isfinite(prices) & (prices > 0)
        if not good.all():
            prices = prices[good]
            dates = [d for d, k in zip(dates, good) if k]

    n = len(prices)
    if n < MIN_TRAIN + 20:
        return None

    lookback = min(LOOKBACK, n - MIN_TRAIN)
    if lookback < 20:
        return None

    initial_train = prices[: n - lookback]
    models = _build_models(initial_train)
    if not models:
        return None
    fns = dict(models)

    # ---- walk-forward backtest (one-step, refit each step) ----
    preds, actuals, prevs = _walk_forward(prices, models, lookback)
    steps = len(actuals)
    half = max(1, steps // 2)
    w_sl, s_sl = slice(0, half), slice(half, steps)  # weight-fit fold, score fold

    # Keep only models that produced a prediction on >=50% of the weight fold
    # (naive is always kept). This drops models silently failing every step.
    kept: List[str] = []
    for name, _ in models:
        arr = np.asarray(preds[name], dtype=float)[w_sl]
        cov = float(np.isfinite(arr).mean()) if len(arr) else 0.0
        if name == "naive" or cov >= 0.5:
            kept.append(name)

    # ---- weights = 1/MAPE, fit on the WEIGHT fold (out-of-sample wrt the score fold) ----
    weights: Dict[str, float] = {}
    for name in kept:
        m = _metrics(np.asarray(preds[name])[w_sl], actuals[w_sl], prevs[w_sl])["mape"]
        weights[name] = (1.0 / max(m, 1e-4)) if (m and m > 0) else (1.0 if name == "naive" else 0.0)
    if sum(weights.values()) <= 0:
        weights = {name: (1.0 if name == "naive" else 0.0) for name in kept}
    tot = sum(weights.values()) or 1.0
    weights = {k: v / tot for k, v in weights.items()}

    # ---- ensemble one-step predictions on the SCORE fold (renormalise per step) ----
    ens_s = np.full(steps - half, np.nan)
    for j, i in enumerate(range(half, steps)):
        avail = [nm for nm in kept if np.isfinite(preds[nm][i]) and weights.get(nm, 0) > 0]
        if not avail:
            continue
        wsum = sum(weights[nm] for nm in avail)
        ens_s[j] = sum(weights[nm] * preds[nm][i] for nm in avail) / wsum

    actuals_s, prevs_s = actuals[s_sl], prevs[s_sl]
    ens_metrics = _metrics(ens_s, actuals_s, prevs_s)
    naive_s = _metrics(np.asarray(preds["naive"])[s_sl], actuals_s, prevs_s)
    naive_mape, ens_mape = naive_s["mape"], ens_metrics["mape"]
    skill = ((naive_mape - ens_mape) / naive_mape) if (naive_mape and ens_mape is not None and naive_mape > 0) else None
    reliable = bool(skill is not None and skill > 0 and ens_metrics["directional"] is not None
                    and ens_metrics["directional"] > 0.5)

    # Per-model metrics for the leaderboard, also reported on the score fold.
    per_model = {nm: _metrics(np.asarray(preds[nm])[s_sl], actuals_s, prevs_s) for nm in kept}

    # ---- multi-step ensemble forecast (kept models + fold-1 weights) ----
    model_forecasts: Dict[str, np.ndarray] = {}
    for name in kept:
        try:
            fc = np.asarray(fns[name](prices, horizon), dtype=float)
            if np.all(np.isfinite(fc)):
                model_forecasts[name] = fc
        except Exception:
            pass
    avail_fc = [nm for nm in model_forecasts if weights.get(nm, 0) > 0]
    if not avail_fc:  # last-resort safety net
        model_forecasts["naive"] = _naive(prices, horizon)
        weights["naive"] = weights.get("naive", 1.0) or 1.0
        avail_fc = ["naive"]
    wsum = sum(weights[nm] for nm in avail_fc) or 1.0
    point = np.zeros(horizon)
    for nm in avail_fc:
        point += (weights[nm] / wsum) * model_forecasts[nm]
    point = np.clip(point, 0.0, None)

    # ---- 80% band: MC bootstrap of ensemble one-step relative errors (score fold) ----
    rel = (actuals_s - ens_s) / np.where(actuals_s != 0, actuals_s, np.nan)
    rel = rel[np.isfinite(rel)]
    rel = np.clip(rel, -0.99, 1.0)  # bound extreme errors so the band stays sensible
    if len(rel) >= 5:
        rng = np.random.default_rng(42)  # fixed seed -> band is reproducible across reruns
        draws = rng.choice(rel, size=(MC_SIMS, horizon), replace=True)
        sim_paths = point[None, :] * np.cumprod(1.0 + draws, axis=1)
        lower = np.clip(np.percentile(sim_paths, 10, axis=0), 0.0, None)
        upper = np.clip(np.percentile(sim_paths, 90, axis=0), 0.0, None)
    else:
        lower, upper = point.copy(), point.copy()

    current_price = float(prices[-1])
    predicted_price = float(point[-1])
    expected_return = (predicted_price - current_price) / current_price if current_price else 0.0
    signal, reasoning = make_signal(expected_return, ens_mape, reliable)

    last_date = dates[-1]
    forecast_dates = [last_date + timedelta(days=h + 1) for h in range(horizon)]

    return {
        "source": source,
        "engine": "Ensemble: " + " + ".join(kept),
        "days": days,
        "horizon": horizon,
        "n_history": n,
        "lookback": lookback,
        "validated_days": steps - half,  # out-of-sample steps the metrics are scored on
        "history_dates": dates,
        "history_prices": [float(p) for p in prices],
        "forecast_dates": forecast_dates,
        "forecast_prices": [float(p) for p in point],
        "lower": [float(x) for x in lower],
        "upper": [float(x) for x in upper],
        "current_price": current_price,
        "predicted_price": predicted_price,
        "expected_return": expected_return,
        "predicted_change": expected_return * 100.0,
        "models": [{"name": nm, **per_model[nm], "weight": weights.get(nm, 0.0)} for nm in kept],
        "ensemble": ens_metrics,
        "naive_mape": naive_mape,
        "skill": skill,
        "reliable": reliable,
        "signal": signal,
        "reasoning": reasoning,
    }
