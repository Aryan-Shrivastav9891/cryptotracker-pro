"""Intraday Signal Lab engine — honest, cost-aware short-horizon backtesting.

Scans a watchlist on 1h/2h horizons, runs the same multi-model ensemble + conformal
machinery as lib/forecast.py on HOURLY data, and ranks coins by their MEASURED,
out-of-sample edge AFTER realistic trading costs (taker fee + slippage + perp
funding). A coin is only "tradeable" if its cost-aware net result beats a naive
baseline out-of-sample. Nothing is hardcoded; every number is backtested.

Accuracy only — not financial advice.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from lib import forecast as F
from lib import history, playbook

# Liquid coins (symbol -> CoinGecko id). Editable in the UI.
WATCHLIST: Dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple", "DOGE": "dogecoin", "ADA": "cardano", "AVAX": "avalanche-2",
    "LINK": "chainlink", "MATIC": "matic-network",
}
HORIZONS = {"1h": 1, "2h": 2}
MIN_TRAIN = 60
WINDOWS = 40           # walk-forward test origins (each refits every model)
MIN_SCORE = 8          # min out-of-sample trades to trust a horizon
# Realistic Binance-perp default costs (fractions).
DEFAULT_COSTS = {"taker_fee": 0.0005, "slippage": 0.0005, "funding_8h": 0.0001}


def _seasonal24(train: np.ndarray, h: int) -> np.ndarray:
    """Daily-seasonal naive for hourly data (period = 24)."""
    s = np.asarray(train, dtype=float)
    if len(s) < 24:
        return F._naive(s, h)
    season = s[-24:]
    return np.array([season[k % 24] for k in range(h)], dtype=float)


def _intra_models(initial_train: np.ndarray, fast: bool = True) -> List[Tuple[str, Callable]]:
    """Model registry (reuses forecast.py model fns). `fast` drops the slow trees/SVR."""
    c: List[Tuple[str, Callable]] = [("naive", F._naive), ("seasonal24", _seasonal24)]
    if F.HAS_STATSMODELS:
        c += [("holt", F._holt), ("theta", F._theta),
              ("arima", F._make_arima(F._best_arima_order(initial_train)))]
    if F.HAS_SKLEARN:
        c += [("hgb", lambda tr, h: F._ml_forecast(tr, h, F._est_hgb)),
              ("ridge", lambda tr, h: F._ml_forecast(tr, h, F._est_ridge)),
              ("elasticnet", lambda tr, h: F._ml_forecast(tr, h, F._est_enet)),
              ("knn", lambda tr, h: F._ml_forecast(tr, h, F._est_knn))]
        if not fast:
            c += [("rf", lambda tr, h: F._ml_forecast(tr, h, F._est_rf)),
                  ("et", lambda tr, h: F._ml_forecast(tr, h, F._est_et)),
                  ("svr", lambda tr, h: F._ml_forecast(tr, h, F._est_svr))]
    if F.HAS_XGB:
        c.append(("xgboost", lambda tr, h: F._ml_forecast(tr, h, F._est_xgb)))
    if F.HAS_LGB:
        c.append(("lightgbm", lambda tr, h: F._ml_forecast(tr, h, F._est_lgb)))
    working = []
    for nm, fn in c:
        try:
            out = fn(initial_train, 1)
            if np.all(np.isfinite(out)) and out[0] > 0:
                working.append((nm, fn))
        except Exception:
            continue
    return working


def simulate_costs(sign: np.ndarray, anchor: np.ndarray, actual: np.ndarray,
                   hold_hours: int, costs: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Cost-aware long/short P&L: gross move in the signal direction minus round-trip
    fees + slippage + funding over the hold. Returns net stats, or None if no trades."""
    sign, anchor, actual = (np.asarray(a, dtype=float) for a in (sign, anchor, actual))
    mask = (np.isfinite(sign) & np.isfinite(anchor) & np.isfinite(actual)
            & (sign != 0) & (anchor > 0))
    if mask.sum() == 0:
        return None
    sgn, a, ac = sign[mask], anchor[mask], actual[mask]
    gross = sgn * (ac / a - 1.0)
    cost = 2.0 * (costs["taker_fee"] + costs["slippage"]) + costs["funding_8h"] * (hold_hours / 8.0)
    net = gross - cost
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq / peak - 1.0)) if len(eq) else 0.0
    wins, losses = net[net > 0], net[net < 0]
    ann = (24.0 / hold_hours) * 365.0
    sharpe = float(np.mean(net) / np.std(net) * np.sqrt(ann)) if np.std(net) > 1e-12 else 0.0
    return {"n": int(mask.sum()), "net_mean": float(np.mean(net)), "hit_rate": float(np.mean(net > 0)),
            "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
            "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
            "maxdd": maxdd, "sharpe": sharpe, "cost_per_trade": float(cost)}


def _eval_intra(model_fc, origins, anchors, prices, names, final_fc, h, costs) -> Optional[Dict[str, Any]]:
    n = len(prices)
    valid = [j for j, t in enumerate(origins) if t + h <= n]
    if len(valid) < 10:
        return None
    preds_by = {nm: np.asarray([model_fc[nm][j][h - 1] for j in valid]) for nm in names}
    actual = np.asarray([prices[origins[j] + h - 1] for j in valid])
    anchor = np.asarray([anchors[j] for j in valid])
    W = len(valid)
    if W >= 18:
        a, b = int(W * 0.45), int(W * 0.70)
        fit_i, sel_i, test_i = list(range(a)), list(range(a, b)), list(range(b, W))
    else:
        a = max(1, W // 2)
        fit_i, test_i = list(range(a)), list(range(a, W))
        sel_i = test_i
    if not fit_i or not test_i:
        return None

    fit_mapes = {nm: F._metrics(preds_by[nm][fit_i], actual[fit_i], anchor[fit_i])["mape"] for nm in names}
    naive_fit = fit_mapes.get("naive")
    kept = [nm for nm in names if nm == "naive"
            or (fit_mapes[nm] is not None and naive_fit is not None and fit_mapes[nm] < naive_fit)]
    if not kept:
        kept = ["naive"]
    weights = F._inv_weights({nm: fit_mapes[nm] for nm in kept})
    coefs = F._stack_fit(preds_by, kept, actual, fit_i)

    combiners = {"inverse-error": lambda idx: F._pred_inverse(preds_by, weights, kept, idx),
                 "median": lambda idx: F._pred_median(preds_by, kept, idx)}
    if coefs:
        combiners["stacking"] = lambda idx: F._pred_stack(preds_by, coefs, idx)
    best, bm = None, np.inf
    for k, fn in combiners.items():
        m = F._metrics(fn(sel_i), actual[sel_i], anchor[sel_i])["mape"]
        if m is not None and m < bm:
            best, bm = k, m
    best = best or "inverse-error"
    chosen = combiners[best]

    test_pred = chosen(test_i)
    tm = F._metrics(test_pred, actual[test_i], anchor[test_i])
    nm_t = F._metrics(preds_by["naive"][test_i], actual[test_i], anchor[test_i])
    skill = ((nm_t["mape"] - tm["mape"]) / nm_t["mape"]
             if (nm_t["mape"] and tm["mape"] is not None and nm_t["mape"] > 0) else None)

    # cost-aware net result on the TEST fold (chosen combiner vs naive)
    net = simulate_costs(np.sign(test_pred - anchor[test_i]), anchor[test_i], actual[test_i], h, costs)
    naive_net = simulate_costs(np.sign(preds_by["naive"][test_i] - anchor[test_i]),
                               anchor[test_i], actual[test_i], h, costs)

    # conformal coverage (calibrate on FIT(+SELECT), measure on TEST)
    calib = fit_i + (sel_i if sel_i is not test_i else [])
    cp = chosen(calib)
    rel = (actual[calib] - cp) / np.where(actual[calib] != 0, actual[calib], np.nan)
    rel = np.clip(rel[np.isfinite(rel)], -0.99, 5.0)
    coverage = None
    if len(rel) >= 5 and len(test_i) >= 3:
        qlo, qhi = np.percentile(rel, 10), np.percentile(rel, 90)
        lo, hi = test_pred * (1 + qlo), test_pred * (1 + qhi)
        ins = [bool(l <= x <= g) for l, g, x, p in zip(lo, hi, actual[test_i], test_pred) if np.isfinite(p)]
        coverage = float(np.mean(ins)) if ins else None

    # forward signal (combiner on full data)
    cur = float(prices[-1])
    path = F._combine_final(best, kept, weights, coefs, final_fc, h)
    fwd = float(path[-1])
    fwd_gross = (fwd - cur) / cur if cur else 0.0
    cost = net["cost_per_trade"] if net else (
        2.0 * (costs["taker_fee"] + costs["slippage"]) + costs["funding_8h"] * (h / 8.0))
    expected_net = abs(fwd_gross) - cost  # net % if we take the model's direction

    tradeable = bool(
        net and naive_net and net["net_mean"] > 0 and net["net_mean"] > naive_net["net_mean"]
        and tm["directional"] is not None and tm["directional"] > 0.5
        and len(test_i) >= MIN_SCORE and skill is not None and skill > 0)
    signal = ("Long" if fwd_gross > 0 else "Short") if tradeable else "No-edge"

    return {
        "horizon": h, "combiner": best, "models_used": [m for m in kept if m != "naive"] or ["naive"],
        "directional": tm["directional"], "mape": tm["mape"], "skill": skill,
        "net_mean": net["net_mean"] if net else None, "naive_net": naive_net["net_mean"] if naive_net else None,
        "hit_rate": net["hit_rate"] if net else None, "sharpe": net["sharpe"] if net else None,
        "maxdd": net["maxdd"] if net else None, "avg_win": net["avg_win"] if net else None,
        "avg_loss": net["avg_loss"] if net else None, "coverage": coverage,
        "expected_net": expected_net, "signal": signal, "tradeable": tradeable,
        "trades": len(test_i),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def scan_coin(symbol: str, coin_id: str, hours: int, costs_tuple: Tuple[float, float, float],
              fast: bool = True) -> Optional[Dict[str, Any]]:
    """Backtest one coin on 1h & 2h with cost-aware net edge. Cached. None if too short."""
    costs = {"taker_fee": costs_tuple[0], "slippage": costs_tuple[1], "funding_8h": costs_tuple[2]}
    dates, closes, source = history.get_hourly_closes(coin_id, symbol, hours=hours)
    prices = np.asarray(closes, dtype=float)
    if len(prices):
        good = np.isfinite(prices) & (prices > 0)
        prices = prices[good]
    n = len(prices)
    if n < MIN_TRAIN + 14:
        return None
    windows = min(WINDOWS, n - MIN_TRAIN - 2)
    if windows < 12:
        return None
    origins = list(range(n - windows, n))
    anchors = np.asarray([prices[t - 1] for t in origins], dtype=float)
    models = _intra_models(prices[: origins[0]], fast=fast)
    if not models:
        return None
    names = [nm for nm, _ in models]
    model_fc = F._walk_forward(prices, models, origins, max(HORIZONS.values()))
    final_fc: Dict[str, np.ndarray] = {}
    for nm, fn in models:
        try:
            fc = np.asarray(fn(prices, max(HORIZONS.values())), dtype=float)
            if np.all(np.isfinite(fc)) and np.all(fc > 0):
                final_fc[nm] = fc
        except Exception:
            pass
    if "naive" not in final_fc:
        final_fc["naive"] = F._naive(prices, max(HORIZONS.values()))

    results, playbooks = {}, {}
    for label, h in HORIZONS.items():
        results[label] = _eval_intra(model_fc, origins, anchors, prices, names, final_fc, h, costs)
        playbooks[label] = playbook.build_playbook(prices, h, costs)
    return {"symbol": symbol, "coin_id": coin_id, "source": source, "n_history": n,
            "current_price": float(prices[-1]), "models_available": names,
            "results": results, "playbooks": playbooks}


def leverage_math(leverage: float, maint_margin: float = 0.005) -> Dict[str, float]:
    """Approx liquidation distance for an isolated position: a move of ~(1/L − mm)
    against you wipes the margin (before fees)."""
    liq = max(1.0 / max(leverage, 1.0) - maint_margin, 0.0)
    return {"liq_move_pct": liq * 100.0, "leverage": leverage, "maint_margin_pct": maint_margin * 100.0}
