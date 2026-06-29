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

HORIZONS = {"1D": 1, "4D": 4, "1W": 7, "1M": 30}  # label -> days ahead
MAX_H = 30             # longest horizon (steps forecast at each backtest origin)
BACKTEST_WINDOWS = 50  # walk-forward origins (each forecasts MAX_H steps)
MIN_TRAIN = 60         # smallest training window (also GBR's minimum)
MIN_SCORE = 5          # min out-of-sample windows required to call a horizon reliable
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


def _walk_forward(prices: np.ndarray, models, origins: List[int], max_h: int) -> Dict[str, np.ndarray]:
    """Expanding-window walk-forward: at each origin t, refit and forecast `max_h`
    steps. Returns {model: array of shape (n_origins, max_h)} with NaN where a
    model failed (so a failure is never scored as a perfect carry-forward)."""
    model_fc: Dict[str, List[np.ndarray]] = {name: [] for name, _ in models}
    for t in origins:
        train = prices[:t]
        for name, fn in models:
            try:
                fc = np.asarray(fn(train, max_h), dtype=float)
            except Exception:
                fc = np.full(max_h, np.nan)
            fc = np.where(np.isfinite(fc) & (fc > 0), fc, np.nan)
            model_fc[name].append(fc)
    return {nm: np.asarray(v, dtype=float) for nm, v in model_fc.items()}


def _fit_weights(preds_by_model: Dict[str, np.ndarray], actuals: np.ndarray,
                 anchors: np.ndarray, kept: List[str]) -> Dict[str, float]:
    """Weights = 1/MAPE (normalised) over the given (weight-fold) data."""
    weights: Dict[str, float] = {}
    for nm in kept:
        m = _metrics(preds_by_model[nm], actuals, anchors)["mape"]
        weights[nm] = (1.0 / max(m, 1e-4)) if (m and m > 0) else (1.0 if nm == "naive" else 0.0)
    if sum(weights.values()) <= 0:
        weights = {nm: (1.0 if nm == "naive" else 0.0) for nm in kept}
    tot = sum(weights.values()) or 1.0
    return {k: v / tot for k, v in weights.items()}


def _ensemble_preds(preds_by_model: Dict[str, np.ndarray], weights: Dict[str, float],
                    kept: List[str]) -> np.ndarray:
    """Per-row weighted ensemble, renormalising over the models available that row."""
    rows = len(next(iter(preds_by_model.values()))) if preds_by_model else 0
    out = np.full(rows, np.nan)
    for i in range(rows):
        avail = [nm for nm in kept if np.isfinite(preds_by_model[nm][i]) and weights.get(nm, 0) > 0]
        if not avail:
            continue
        wsum = sum(weights[nm] for nm in avail)
        out[i] = sum(weights[nm] * preds_by_model[nm][i] for nm in avail) / wsum
    return out


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
def forecast_coin(coin_id: str, symbol: str, days: int = 540) -> Optional[Dict[str, Any]]:
    """Backtest + ensemble forecast for one coin across ALL horizons (1D/4D/1W/1M).

    Returns a result dict (or None when there isn't enough history). Each horizon
    is validated out-of-sample independently and flagged reliable/unreliable.
    """
    dates, closes, source = history.get_daily_closes(coin_id, symbol, days=days)
    prices = np.asarray(closes, dtype=float)

    # Drop any non-finite / non-positive points (rare bad data); keep dates aligned.
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

    initial_train = prices[: origins[0]]
    models = _build_models(initial_train)
    if not models:
        return None
    fns = dict(models)
    names = [nm for nm, _ in models]

    # ---- one walk-forward producing MAX_H-step forecasts at every origin ----
    model_fc = _walk_forward(prices, models, origins, MAX_H)

    # ---- final forecasts on the full series (one per model, MAX_H steps) ----
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

    current_price = float(prices[-1])
    last_date = dates[-1]

    horizons_out: List[Dict[str, Any]] = []
    for label, h in HORIZONS.items():
        # Valid origins for this horizon (actual price h steps ahead must exist).
        valid = [j for j, t in enumerate(origins) if t + h <= n]
        if len(valid) < 4:
            horizons_out.append(_unreliable_horizon(label, h, current_price, last_date,
                                                    final_fc, names, dates, prices))
            continue

        # h-step preds / actuals / anchors for the valid origins.
        preds_by = {nm: np.asarray([model_fc[nm][j][h - 1] for j in valid]) for nm in names}
        actual = np.asarray([prices[origins[j] + h - 1] for j in valid])
        anchor = np.asarray([anchors[j] for j in valid])

        W = len(valid)
        half = max(1, W // 2)
        wi, si = list(range(half)), list(range(half, W))

        kept = [nm for nm in names
                if nm == "naive" or float(np.isfinite(preds_by[nm][wi]).mean()) >= 0.5]
        weights = _fit_weights({nm: preds_by[nm][wi] for nm in kept},
                               actual[wi], anchor[wi], kept)

        ens_s = _ensemble_preds({nm: preds_by[nm][si] for nm in kept}, weights, kept)
        actual_s, anchor_s = actual[si], anchor[si]
        ens_m = _metrics(ens_s, actual_s, anchor_s)
        naive_m = _metrics(preds_by["naive"][si], actual_s, anchor_s)
        skill = ((naive_m["mape"] - ens_m["mape"]) / naive_m["mape"]
                 if (naive_m["mape"] and ens_m["mape"] is not None and naive_m["mape"] > 0) else None)
        reliable = bool(skill is not None and skill > 0 and ens_m["directional"] is not None
                        and ens_m["directional"] > 0.5 and len(si) >= MIN_SCORE)

        # This horizon's OWN out-of-sample relative errors (no i.i.d. 1-step assumption).
        rel_h = (actual_s - ens_s) / np.where(actual_s != 0, actual_s, np.nan)
        rel_h = np.clip(rel_h[np.isfinite(rel_h)], -0.99, 5.0)

        # Final ensemble path for this horizon (full-data forecasts, horizon weights).
        avail = [nm for nm in kept if nm in final_fc and weights.get(nm, 0) > 0]
        if not avail:
            avail, weights = ["naive"], {**weights, "naive": 1.0}
        wsum = sum(weights[nm] for nm in avail) or 1.0
        path = np.zeros(h)
        for nm in avail:
            path += (weights[nm] / wsum) * final_fc[nm][:h]
        path = np.clip(path, 0.0, None)

        lower, upper = _band(path, rel_h)
        predicted_price = float(path[-1])
        exp_ret = (predicted_price - current_price) / current_price if current_price else 0.0
        signal, reasoning = make_signal(exp_ret, ens_m["mape"], reliable)

        horizons_out.append({
            "label": label, "h": h,
            "predicted_price": predicted_price,
            "expected_return": exp_ret, "predicted_change": exp_ret * 100.0,
            "forecast_path": [float(x) for x in path],
            "forecast_dates": [last_date + timedelta(days=k + 1) for k in range(h)],
            "lower": [float(x) for x in lower], "upper": [float(x) for x in upper],
            "mape": ens_m["mape"], "mae": ens_m["mae"], "rmse": ens_m["rmse"],
            "directional": ens_m["directional"], "naive_mape": naive_m["mape"],
            "skill": skill, "reliable": reliable, "signal": signal, "reasoning": reasoning,
            "windows": W, "score_windows": len(si),
        })

    return {
        "source": source,
        "engine": "Ensemble: " + " + ".join(names),
        "days": days,
        "n_history": n,
        "current_price": current_price,
        "history_dates": dates,
        "history_prices": [float(p) for p in prices],
        "horizons": horizons_out,
    }


def _band(path: np.ndarray, rel_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """80% band from the horizon's OWN h-step relative errors (10th/90th pct),
    widened from step 1 to the horizon end via sqrt-time scaling. Uses each
    horizon's measured error directly — no i.i.d. one-step compounding assumption."""
    h = len(path)
    if rel_h is None or len(rel_h) < 5:
        return path.copy(), path.copy()
    p10, p90 = float(np.percentile(rel_h, 10)), float(np.percentile(rel_h, 90))
    lower, upper = np.empty(h), np.empty(h)
    for k in range(1, h + 1):
        scale = np.sqrt(k / h)  # grows to the full measured h-step error at the end
        lower[k - 1] = max(0.0, path[k - 1] * (1.0 + p10 * scale))
        upper[k - 1] = max(0.0, path[k - 1] * (1.0 + p90 * scale))
    return lower, upper


def _unreliable_horizon(label, h, current_price, last_date, final_fc, names, dates, prices):
    """Produce a point forecast for a horizon we couldn't backtest (history too short)."""
    path = np.clip(final_fc.get("naive", _naive(prices, h))[:h], 0.0, None)
    predicted_price = float(path[-1]) if len(path) else current_price
    exp_ret = (predicted_price - current_price) / current_price if current_price else 0.0
    return {
        "label": label, "h": h,
        "predicted_price": predicted_price,
        "expected_return": exp_ret, "predicted_change": exp_ret * 100.0,
        "forecast_path": [float(x) for x in path],
        "forecast_dates": [last_date + timedelta(days=k + 1) for k in range(h)],
        "lower": [float(x) for x in path], "upper": [float(x) for x in path],
        "mape": None, "mae": None, "rmse": None, "directional": None, "naive_mape": None,
        "skill": None, "reliable": False,
        "signal": "Hold", "reasoning": "Not enough history to backtest this horizon — Hold.",
        "windows": 0, "score_windows": 0,
    }
