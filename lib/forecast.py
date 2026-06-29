"""Robust multi-model, multi-horizon forecasting with HONEST, calibrated uncertainty.

Principle: more algorithms does NOT mean beating the market. Every horizon's final
ensemble is measured OUT-OF-SAMPLE against a naive baseline; if it has no edge the
UI says so. We never fake confidence.

Per coin, per horizon (1D / 4D / 1W / 1M):
  1. Pull ~1.5–2y daily closes (lib.history: Binance via ccxt, else CoinGecko).
  2. Run many base models in a walk-forward backtest (expanding window, h-step-ahead,
     refit each origin). Optional deps (xgboost, lightgbm, prophet, statsforecast,
     tensorflow, arch) degrade silently — a model that errors/NaNs is skipped.
  3. DYNAMIC SELECTION: drop models whose backtest MAPE is worse than the naive baseline.
  4. THREE combiners — inverse-error weighted, robust median, and stacking (NNLS on the
     base models' out-of-fold predictions). Pick the best per coin+horizon on a held-out
     selection fold; report which combiner + which models won.
  5. CONFORMAL 80% band from the chosen combiner's residual quantiles (split-conformal),
     with the ACTUAL measured coverage shown. Optional GARCH(1,1) shapes the band width
     across longer horizons.
Not financial advice.
"""
from __future__ import annotations

import warnings
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from lib import features, history

# ---- optional libraries (all guarded; missing ones are simply skipped) ------
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

try:
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  RandomForestRegressor, ExtraTreesRegressor)
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

try:
    from scipy.optimize import nnls
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:  # pragma: no cover
    HAS_PROPHET = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:  # pragma: no cover
    HAS_ARCH = False

HORIZONS = {"1D": 1, "4D": 4, "1W": 7, "1M": 30}
MAX_H = 30
BACKTEST_WINDOWS = 36   # walk-forward origins (each refits every model, forecasts MAX_H)
MIN_TRAIN = 60
MIN_SCORE = 5
MAX_PCT_ERROR = 5.0
RET_CLIP = 0.25
SEASON = 7
COVERAGE_TARGET = 0.80


# =========================================================================== #
# Base forecasters: fn(train: np.ndarray, h: int) -> np.ndarray of h prices
# =========================================================================== #
def _naive(train: np.ndarray, h: int) -> np.ndarray:
    """Random-walk-with-drift (the baseline everything must beat)."""
    s = np.asarray(train, dtype=float)
    pos = s[s > 0]
    if len(pos) < 2:
        last = float(s[-1]) if len(s) else 0.0
        return np.full(h, last)
    mu = float(np.mean(np.diff(np.log(pos))))
    return float(pos[-1]) * np.exp(mu * np.arange(1, h + 1))


def _seasonal_naive(train: np.ndarray, h: int) -> np.ndarray:
    """Weekly seasonal naive: repeat the value from one 7-day season ago."""
    s = np.asarray(train, dtype=float)
    if len(s) < SEASON:
        return _naive(s, h)
    season = s[-SEASON:]
    return np.array([season[k % SEASON] for k in range(h)], dtype=float)


def _holt(train: np.ndarray, h: int) -> np.ndarray:
    s = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(np.log(s), trend="add", damped_trend=True,
                                   initialization_method="estimated").fit(optimized=True)
        fc = np.exp(np.asarray(fit.forecast(h), dtype=float))
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite holt")
    return np.clip(fc, 0.0, None)


def _theta(train: np.ndarray, h: int) -> np.ndarray:
    s = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = np.asarray(ThetaModel(s, period=SEASON, deseasonalize=False).fit().forecast(h), dtype=float)
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite theta")
    return np.clip(fc, 0.0, None)


def _make_arima(order: Tuple[int, int, int]) -> Callable[[np.ndarray, int], np.ndarray]:
    def _arima(train: np.ndarray, h: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(np.log(np.asarray(train, dtype=float)), order=order).fit()
            fc = np.exp(np.asarray(fit.forecast(h), dtype=float))
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite arima")
        return np.clip(fc, 0.0, None)
    return _arima


def _best_arima_order(train: np.ndarray) -> Tuple[int, int, int]:
    arr = np.asarray(train, dtype=float)
    arr = arr[arr > 0]
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


def _sarima(train: np.ndarray, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(np.log(np.asarray(train, dtype=float)), order=(1, 1, 1),
                      seasonal_order=(1, 0, 1, SEASON), enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        fc = np.exp(np.asarray(fit.forecast(h), dtype=float))
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite sarima")
    return np.clip(fc, 0.0, None)


def _uc(train: np.ndarray, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = UnobservedComponents(np.log(np.asarray(train, dtype=float)),
                                   level="local linear trend").fit(disp=False)
        fc = np.exp(np.asarray(fit.forecast(h), dtype=float))
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite uc")
    return np.clip(fc, 0.0, None)


# ---- ML: predict next-step log return from features, reconstruct path -------
def _ml_forecast(train: np.ndarray, h: int, make_estimator: Callable) -> np.ndarray:
    if len(train) < MIN_TRAIN:
        raise RuntimeError("ml: short train")
    X, y = features.build_xy(train)
    if len(X) < 30:
        raise RuntimeError("ml: few rows")
    est = make_estimator()
    est.fit(X, y)
    series = list(map(float, train))
    out: List[float] = []
    for _ in range(h):
        row = features.latest_row(np.asarray(series[-80:])).reshape(1, -1)
        r = 0.0 if not np.all(np.isfinite(row)) else float(est.predict(row)[0])
        r = float(np.clip(r, -RET_CLIP, RET_CLIP))
        nxt = series[-1] * np.exp(r)
        out.append(nxt)
        series.append(nxt)
    return np.asarray(out, dtype=float)


def _est_hgb():
    return HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.06,
                                         l2_regularization=1.0, min_samples_leaf=20, random_state=42)


def _est_rf():
    return RandomForestRegressor(n_estimators=30, max_depth=4, min_samples_leaf=12,
                                 n_jobs=-1, random_state=42)


def _est_et():
    return ExtraTreesRegressor(n_estimators=30, max_depth=4, min_samples_leaf=12,
                               n_jobs=-1, random_state=42)


def _est_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _est_enet():
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000))


def _est_knn():
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))


def _est_svr():
    return make_pipeline(StandardScaler(), SVR(C=1.0, gamma="scale", epsilon=0.001))


def _est_xgb():
    return XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.06,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)


def _est_lgb():
    return LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.06,
                         num_leaves=15, random_state=42, verbose=-1)


def _prophet(train: np.ndarray, h: int) -> np.ndarray:  # pragma: no cover (optional)
    df = pd.DataFrame({"ds": pd.date_range("2000-01-01", periods=len(train), freq="D"),
                       "y": np.log(np.asarray(train, dtype=float))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=h)
        fc = np.exp(m.predict(future)["yhat"].to_numpy()[-h:])
    if not np.all(np.isfinite(fc)):
        raise ValueError("non-finite prophet")
    return np.clip(fc, 0.0, None)


# =========================================================================== #
# Model registry
# =========================================================================== #
def _build_models(initial_train: np.ndarray) -> List[Tuple[str, Callable]]:
    """Assemble candidate models, validated on the initial train (broken ones dropped)."""
    cands: List[Tuple[str, Callable]] = [("naive", _naive), ("seasonal_naive", _seasonal_naive)]
    if HAS_STATSMODELS:
        cands += [("holt", _holt), ("theta", _theta),
                  ("arima", _make_arima(_best_arima_order(initial_train))),
                  ("sarima", _sarima), ("uc", _uc)]
    if HAS_SKLEARN:
        for nm, fac in [("hgb", _est_hgb), ("rf", _est_rf), ("et", _est_et),
                        ("ridge", _est_ridge), ("elasticnet", _est_enet),
                        ("knn", _est_knn), ("svr", _est_svr)]:
            cands.append((nm, (lambda tr, h, f=fac: _ml_forecast(tr, h, f))))
    if HAS_XGB:
        cands.append(("xgboost", (lambda tr, h: _ml_forecast(tr, h, _est_xgb))))
    if HAS_LGB:
        cands.append(("lightgbm", (lambda tr, h: _ml_forecast(tr, h, _est_lgb))))
    if HAS_PROPHET:
        cands.append(("prophet", _prophet))

    working: List[Tuple[str, Callable]] = []
    for name, fn in cands:
        try:
            out = fn(initial_train, 1)
            if np.all(np.isfinite(out)) and out[0] > 0:
                working.append((name, fn))
        except Exception:
            continue
    return working


def _walk_forward(prices: np.ndarray, models, origins: List[int], max_h: int) -> Dict[str, np.ndarray]:
    """At each origin t: refit every model on prices[:t], forecast max_h steps.
    Returns {model: (n_origins, max_h)} with NaN where a model failed."""
    model_fc: Dict[str, List[np.ndarray]] = {name: [] for name, _ in models}
    for t in origins:
        train = prices[:t]
        for name, fn in models:
            try:
                fc = np.asarray(fn(train, max_h), dtype=float)
            except Exception:
                fc = np.full(max_h, np.nan)
            model_fc[name].append(np.where(np.isfinite(fc) & (fc > 0), fc, np.nan))
    return {nm: np.asarray(v, dtype=float) for nm, v in model_fc.items()}


# =========================================================================== #
# Metrics + combiners
# =========================================================================== #
def _metrics(preds: np.ndarray, actuals: np.ndarray, prevs: np.ndarray) -> Dict[str, Optional[float]]:
    preds, actuals, prevs = (np.asarray(a, dtype=float) for a in (preds, actuals, prevs))
    mask = np.isfinite(preds) & np.isfinite(actuals)
    if not mask.any():
        return {"mape": None, "mae": None, "rmse": None, "directional": None}
    p, a, pv = preds[mask], actuals[mask], prevs[mask]
    err = p - a
    nz = a != 0
    mape = float(np.mean(np.minimum(np.abs(err[nz]) / np.abs(a[nz]), MAX_PCT_ERROR))) if nz.any() else None
    moved = a != pv
    directional = (float(np.mean(np.sign(p[moved] - pv[moved]) == np.sign(a[moved] - pv[moved])))
                   if moved.any() else None)
    return {"mape": mape, "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "directional": directional}


def _inv_weights(mapes: Dict[str, Optional[float]]) -> Dict[str, float]:
    w = {nm: (1.0 / max(m, 1e-4)) if (m and m > 0) else 0.0 for nm, m in mapes.items()}
    if sum(w.values()) <= 0:
        w = {nm: 1.0 for nm in mapes}
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def _pred_inverse(preds_by, weights, kept, idx) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        avail = [nm for nm in kept if np.isfinite(preds_by[nm][i]) and weights.get(nm, 0) > 0]
        if avail:
            wsum = sum(weights[nm] for nm in avail)
            out[j] = sum(weights[nm] * preds_by[nm][i] for nm in avail) / wsum
    return out


def _pred_median(preds_by, kept, idx) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        vals = [preds_by[nm][i] for nm in kept if np.isfinite(preds_by[nm][i])]
        if vals:
            out[j] = float(np.median(vals))
    return out


def _stack_fit(preds_by, kept, actual, idx) -> Optional[Dict[str, float]]:
    if not HAS_SCIPY or len(kept) < 2:
        return None
    rows = [i for i in idx if all(np.isfinite(preds_by[nm][i]) for nm in kept)]
    if len(rows) < max(6, len(kept) + 2):
        return None
    A = np.array([[preds_by[nm][i] for nm in kept] for i in rows], dtype=float)
    b = np.array([actual[i] for i in rows], dtype=float)
    try:
        coef, _ = nnls(A, b)
    except Exception:
        return None
    if not np.isfinite(coef).all() or coef.sum() <= 0:
        return None
    return {nm: float(c) for nm, c in zip(kept, coef)}


def _pred_stack(preds_by, coefs, idx) -> np.ndarray:
    total = sum(coefs.values()) or 1.0
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        avail = [nm for nm in coefs if np.isfinite(preds_by[nm][i])]
        ac = sum(coefs[nm] for nm in avail)
        if avail and ac > 0:
            out[j] = sum(coefs[nm] * preds_by[nm][i] for nm in avail) * (total / ac)
    return out


def _combine_final(kind, kept, weights, coefs, final_fc, h) -> np.ndarray:
    """Build the forward multi-step path from full-data model forecasts."""
    avail = [nm for nm in kept if nm in final_fc]
    if not avail:
        avail = ["naive"] if "naive" in final_fc else list(final_fc)[:1]
    paths = {nm: np.clip(final_fc[nm][:h], 0.0, None) for nm in avail}
    if kind == "median":
        return np.median(np.vstack([paths[nm] for nm in avail]), axis=0)
    if kind == "stacking" and coefs:
        cs = {nm: coefs.get(nm, 0.0) for nm in avail}
        total = sum(coefs.values()) or 1.0
        ac = sum(cs.values())
        if ac > 0:
            return np.clip(sum(cs[nm] * paths[nm] for nm in avail) * (total / ac), 0.0, None)
    # default: inverse-error weighted
    wsum = sum(weights.get(nm, 0) for nm in avail) or 1.0
    if wsum <= 0:
        return np.mean(np.vstack([paths[nm] for nm in avail]), axis=0)
    return np.clip(sum(weights.get(nm, 0) * paths[nm] for nm in avail) / wsum, 0.0, None)


# =========================================================================== #
# Conformal band + GARCH width
# =========================================================================== #
def _garch_shape(train_prices: np.ndarray, h: int) -> np.ndarray:
    """Per-step band-width multiplier (normalised to 1 at the final step).

    Uses a GARCH(1,1) volatility term structure if `arch` is installed; otherwise
    falls back to sqrt-time widening."""
    fallback = np.sqrt(np.arange(1, h + 1) / h)
    if not HAS_ARCH or h <= 1:
        return fallback
    try:  # pragma: no cover (optional dependency)
        r = np.diff(np.log(np.asarray(train_prices, dtype=float))) * 100.0
        res = arch_model(r, vol="Garch", p=1, q=1, mean="Zero").fit(disp="off")
        var = res.forecast(horizon=h, reindex=False).variance.values[-1]
        cum = np.sqrt(np.cumsum(var))
        return cum / cum[-1] if cum[-1] > 0 else fallback
    except Exception:
        return fallback


def _conformal_band(path: np.ndarray, rel_resid: np.ndarray, train_prices: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Split-conformal 80% band: the chosen combiner's signed relative residuals give
    the 10th/90th-percentile offsets at the horizon end; GARCH/sqrt shapes the width
    across intermediate steps."""
    h = len(path)
    if rel_resid is None or len(rel_resid) < 5:
        return path.copy(), path.copy()
    q_lo = float(np.percentile(rel_resid, 100 * (1 - COVERAGE_TARGET) / 2))      # ~10th
    q_hi = float(np.percentile(rel_resid, 100 * (1 + COVERAGE_TARGET) / 2))      # ~90th
    shape = _garch_shape(train_prices, h)
    lower = np.maximum(0.0, path * (1.0 + q_lo * shape))
    upper = np.maximum(0.0, path * (1.0 + q_hi * shape))
    return lower, upper


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
        return "Strong Buy", f"Expected {er:+.1%} over the horizon vs typical error ±{thr:.1%}."
    if er >= thr:
        return "Buy", f"Expected {er:+.1%} over the horizon vs typical error ±{thr:.1%}."
    if er <= -2 * thr:
        return "Strong Sell", f"Expected {er:+.1%} over the horizon vs typical error ±{thr:.1%}."
    if er <= -thr:
        return "Sell", f"Expected {er:+.1%} over the horizon vs typical error ±{thr:.1%}."
    return "Hold", f"Expected move {er:+.1%} is within the model's typical error (±{thr:.1%}) — Hold."


def tilt_signal(signal: str, sentiment_label: str) -> str:
    if signal not in _SIG_ORDER:
        return signal
    i = _SIG_ORDER.index(signal)
    if sentiment_label == "Bullish":
        i = min(i + 1, len(_SIG_ORDER) - 1)
    elif sentiment_label == "Bearish":
        i = max(i - 1, 0)
    return _SIG_ORDER[i]


# =========================================================================== #
# Per-horizon evaluation
# =========================================================================== #
def _eval_horizon(label, h, model_fc, origins, anchors, prices, dates, names, final_fc):
    n = len(prices)
    current = float(prices[-1])
    last_date = dates[-1]
    valid = [j for j, t in enumerate(origins) if t + h <= n]
    if len(valid) < 6:
        return _unreliable_horizon(label, h, current, last_date, final_fc, prices)

    preds_by = {nm: np.asarray([model_fc[nm][j][h - 1] for j in valid]) for nm in names}
    actual = np.asarray([prices[origins[j] + h - 1] for j in valid])
    anchor = np.asarray([anchors[j] for j in valid])
    W = len(valid)

    # Folds: FIT (params) / SELECT (pick combiner) / TEST (report). 3-way when enough data.
    if W >= 15:
        a, b = int(W * 0.4), int(W * 0.7)
        fit_i, sel_i, test_i = list(range(a)), list(range(a, b)), list(range(b, W))
        select_on_test = False
    else:
        a = max(1, W // 2)
        fit_i, test_i = list(range(a)), list(range(a, W))
        sel_i = test_i
        select_on_test = True
    if not test_i or not fit_i:
        return _unreliable_horizon(label, h, current, last_date, final_fc, prices)

    # Dynamic selection: keep models that beat naive's MAPE on the FIT fold (+naive).
    fit_mapes = {nm: _metrics(preds_by[nm][fit_i], actual[fit_i], anchor[fit_i])["mape"] for nm in names}
    naive_fit = fit_mapes.get("naive")
    kept = [nm for nm in names if nm == "naive"
            or (fit_mapes[nm] is not None and naive_fit is not None and fit_mapes[nm] < naive_fit)]
    if not kept:
        kept = ["naive"]

    weights = _inv_weights({nm: fit_mapes[nm] for nm in kept})
    coefs = _stack_fit(preds_by, kept, actual, fit_i)

    combiners = {
        "inverse-error": lambda idx: _pred_inverse(preds_by, weights, kept, idx),
        "median": lambda idx: _pred_median(preds_by, kept, idx),
    }
    if coefs:
        combiners["stacking"] = lambda idx: _pred_stack(preds_by, coefs, idx)

    # Pick the combiner with the lowest MAPE on the SELECT fold.
    best_kind, best_mape = None, np.inf
    for kind, fn in combiners.items():
        m = _metrics(fn(sel_i), actual[sel_i], anchor[sel_i])["mape"]
        if m is not None and m < best_mape:
            best_kind, best_mape = kind, m
    if best_kind is None:
        best_kind = "inverse-error"

    chosen = combiners[best_kind]

    # Report metrics on the TEST fold (held out from params).
    test_pred = chosen(test_i)
    tm = _metrics(test_pred, actual[test_i], anchor[test_i])
    naive_test = _metrics(preds_by["naive"][test_i], actual[test_i], anchor[test_i])
    skill = ((naive_test["mape"] - tm["mape"]) / naive_test["mape"]
             if (naive_test["mape"] and tm["mape"] is not None and naive_test["mape"] > 0) else None)
    reliable = bool(skill is not None and skill > 0 and tm["directional"] is not None
                    and tm["directional"] > 0.5 and len(test_i) >= MIN_SCORE)

    # Conformal calibration on FIT+SELECT residuals; measure coverage on TEST.
    calib_idx = fit_i + (sel_i if not select_on_test else [])
    calib_pred = chosen(calib_idx)
    rel_cal = (actual[calib_idx] - calib_pred) / np.where(actual[calib_idx] != 0, actual[calib_idx], np.nan)
    rel_cal = np.clip(rel_cal[np.isfinite(rel_cal)], -0.99, 5.0)
    coverage = None
    if len(rel_cal) >= 5 and len(test_i) >= 3:
        q_lo = float(np.percentile(rel_cal, 100 * (1 - COVERAGE_TARGET) / 2))
        q_hi = float(np.percentile(rel_cal, 100 * (1 + COVERAGE_TARGET) / 2))
        lo_t = test_pred * (1.0 + q_lo)
        hi_t = test_pred * (1.0 + q_hi)
        inside = [(lo <= a <= hi) for lo, hi, a, p in zip(lo_t, hi_t, actual[test_i], test_pred)
                  if np.isfinite(p)]
        coverage = float(np.mean(inside)) if inside else None

    # Final forward forecast: use the SAME combiner params chosen on FIT (do NOT refit on
    # all windows) so (a) the forward path never sees the TEST fold, and (b) the conformal
    # band residuals (from `chosen`) match the exact params that built the path.
    all_i = list(range(W))
    path = _combine_final(best_kind, kept, weights, coefs, final_fc, h)

    rel_all = (actual - chosen(all_i)) / np.where(actual != 0, actual, np.nan)
    rel_all = np.clip(rel_all[np.isfinite(rel_all)], -0.99, 5.0)
    lower, upper = _conformal_band(path, rel_all, prices)

    predicted_price = float(path[-1])
    exp_ret = (predicted_price - current) / current if current else 0.0
    signal, reasoning = make_signal(exp_ret, tm["mape"], reliable)
    models_used = [nm for nm in kept if nm != "naive" or len(kept) == 1]

    return {
        "label": label, "h": h,
        "predicted_price": predicted_price,
        "expected_return": exp_ret, "predicted_change": exp_ret * 100.0,
        "forecast_path": [float(x) for x in path],
        "forecast_dates": [last_date + timedelta(days=k + 1) for k in range(h)],
        "lower": [float(x) for x in lower], "upper": [float(x) for x in upper],
        "mape": tm["mape"], "mae": tm["mae"], "rmse": tm["rmse"],
        "directional": tm["directional"], "naive_mape": naive_test["mape"],
        "skill": skill, "reliable": reliable, "signal": signal, "reasoning": reasoning,
        "combiner": best_kind, "models_used": models_used, "n_models": len(kept),
        "coverage": coverage, "select_on_test": select_on_test,
        "windows": W, "score_windows": len(test_i),
    }


def _unreliable_horizon(label, h, current, last_date, final_fc, prices):
    path = np.clip(final_fc.get("naive", _naive(prices, h))[:h], 0.0, None)
    predicted_price = float(path[-1]) if len(path) else current
    exp_ret = (predicted_price - current) / current if current else 0.0
    return {
        "label": label, "h": h,
        "predicted_price": predicted_price,
        "expected_return": exp_ret, "predicted_change": exp_ret * 100.0,
        "forecast_path": [float(x) for x in path],
        "forecast_dates": [last_date + timedelta(days=k + 1) for k in range(h)],
        "lower": [float(x) for x in path], "upper": [float(x) for x in path],
        "mape": None, "mae": None, "rmse": None, "directional": None, "naive_mape": None,
        "skill": None, "reliable": False, "signal": "Hold",
        "reasoning": "Not enough history to backtest this horizon — Hold.",
        "combiner": None, "models_used": [], "n_models": 0, "coverage": None,
        "select_on_test": False, "windows": 0, "score_windows": 0,
    }


# =========================================================================== #
# Public entry point
# =========================================================================== #
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_coin(coin_id: str, symbol: str, days: int = 540) -> Optional[Dict[str, Any]]:
    """Multi-model, multi-horizon backtest + forecast for one coin. None if too short."""
    dates, closes, source = history.get_daily_closes(coin_id, symbol, days=days)
    prices = np.asarray(closes, dtype=float)
    if len(prices):
        good = np.isfinite(prices) & (prices > 0)
        if not good.all():
            prices = prices[good]
            dates = [d for d, k in zip(dates, good) if k]

    n = len(prices)
    if n < MIN_TRAIN + 12:
        return None
    windows = min(BACKTEST_WINDOWS, n - MIN_TRAIN)
    if windows < 10:
        return None

    origins = list(range(n - windows, n))
    anchors = np.asarray([prices[t - 1] for t in origins], dtype=float)
    models = _build_models(prices[: origins[0]])
    if not models:
        return None
    names = [nm for nm, _ in models]

    model_fc = _walk_forward(prices, models, origins, MAX_H)

    final_fc: Dict[str, np.ndarray] = {}
    for nm, fn in models:
        try:
            fc = np.asarray(fn(prices, MAX_H), dtype=float)
            if np.all(np.isfinite(fc)) and np.all(fc > 0):
                final_fc[nm] = fc
        except Exception:
            pass
    if "naive" not in final_fc:
        final_fc["naive"] = _naive(prices, MAX_H)

    horizons_out = [_eval_horizon(label, h, model_fc, origins, anchors, prices, dates, names, final_fc)
                    for label, h in HORIZONS.items()]

    return {
        "source": source,
        "engine": f"{len(names)}-model ensemble: " + ", ".join(names),
        "models_available": names,
        "days": days,
        "n_history": n,
        "current_price": float(prices[-1]),
        "history_dates": dates,
        "history_prices": [float(p) for p in prices],
        "horizons": horizons_out,
    }
