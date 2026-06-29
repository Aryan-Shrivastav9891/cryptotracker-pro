"""Entry Playbook — regime-driven dip vs breakout setups, honestly backtested.

Educational only — NOT financial advice. For one coin + horizon:
  1. Classify the current regime (Trend vs Range) from leakage-free features.
  2. Define two entry STYLES from past-only signals:
       • Dip / mean-reversion  — enter when price is stretched from its mean (fits Range).
       • Breakout / momentum   — enter when price breaks a recent high/low (fits Trend).
  3. Label every historical trigger with the TRIPLE-BARRIER method (TP at +k·ATR,
     SL at −k·ATR, time barrier = horizon h) and measure the cost-aware outcome.
  4. Evaluate the regime-fitting style on a PURGED time split: it only "qualifies"
     if its out-of-sample expectancy is positive AND beats a cost-aware buy-&-hold
     baseline. Otherwise -> No-trade.
  5. A META-LABEL logistic model gives the calibrated P(TP before SL) for the
     current setup; below a threshold -> Skip.
  6. Suggested (illustrative) entry zone / stop / target come from current price + ATR.

Every number is measured out-of-sample, net of costs. Levels are illustrative, never
instructions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from lib import features

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

K_SL = 1.5          # stop-loss barrier in ATR units
K_TP = 2.0          # take-profit barrier in ATR units (payoff ratio = K_TP/K_SL)
REGIME_TREND = 0.15  # |SMA-slope|/volatility above this = Trend, else Range
MIN_TRADES = 15     # min historical setups to evaluate a style at all
MIN_TEST = 8        # min out-of-sample setups to trust the edge
PROB_THRESHOLD = 0.50
_FEAT_COLS = ["boll_z", "rsi14", "sma10_slope", "sma20_slope", "regime",
              "vol10", "vol20", "macd", "r1", "r2"]


def _atr_pct(prices: np.ndarray, win: int = 14) -> np.ndarray:
    close = pd.Series(prices)
    atr = close.diff().abs().rolling(win).mean()
    return (atr / close).to_numpy()


def _eval_style(prices: np.ndarray, side: np.ndarray, feat_mat: np.ndarray,
                h: int, cost: float, atr_pct: np.ndarray) -> Dict[str, Any]:
    """Triple-barrier backtest of one entry style on a purged time split."""
    n = len(prices)
    trades: List[tuple] = []  # (bar_index, side, net_return, tp_first_label, features)
    for i in range(30, n - 1):
        s = side[i]
        if s == 0 or not np.isfinite(atr_pct[i]) or atr_pct[i] <= 0:
            continue
        if not np.all(np.isfinite(feat_mat[i])):
            continue
        entry, atrp = prices[i], atr_pct[i]
        tp = entry * (1 + s * K_TP * atrp)
        sl = entry * (1 - s * K_SL * atrp)
        end = min(i + h, n - 1)
        out, label = None, None
        for j in range(i + 1, end + 1):
            p = prices[j]
            if s > 0:
                if p >= tp:
                    out, label = tp, 1; break
                if p <= sl:
                    out, label = sl, 0; break
            else:
                if p <= tp:
                    out, label = tp, 1; break
                if p >= sl:
                    out, label = sl, 0; break
        if out is None:  # time barrier
            out = prices[end]
            label = 1 if (s * (out / entry - 1)) > 0 else 0
        net = s * (out / entry - 1) - cost
        trades.append((i, s, net, label, feat_mat[i]))

    if len(trades) < MIN_TRADES:
        return {"has_edge": False, "n": len(trades), "reason": "too few setups"}

    idx = np.array([t[0] for t in trades])
    rets = np.array([t[2] for t in trades], dtype=float)
    labels = np.array([t[3] for t in trades], dtype=int)
    X = np.array([t[4] for t in trades], dtype=float)

    cut = int(len(trades) * 0.6)
    test_start_bar = idx[cut]
    # purge train setups whose label window overlaps the test start (embargo = h bars)
    pos = np.arange(len(trades))
    train = (pos < cut) & ~(idx + h >= test_start_bar)
    test = pos >= cut
    if train.sum() < MIN_TEST or test.sum() < MIN_TEST:
        return {"has_edge": False, "n": len(trades), "reason": "too few after purge"}

    exp_test = float(np.mean(rets[test]))
    win_test = float(np.mean(labels[test]))
    # cost-aware naive baseline: buy & hold for h bars, entered at every test-period bar
    bars = np.arange(test_start_bar, n - h)
    bh = float(np.mean(prices[bars + h] / prices[bars] - 1.0) - cost) if len(bars) else 0.0
    has_edge = bool(exp_test > 0 and exp_test > bh and test.sum() >= MIN_TEST)

    # meta-label: calibrated P(TP before SL) for the current setup
    prob_current = None
    if _HAS_SK and len(set(labels[train].tolist())) > 1:
        try:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            clf.fit(X[train], labels[train])
            cur = feat_mat[-1]
            if np.all(np.isfinite(cur)):
                prob_current = float(clf.predict_proba(cur.reshape(1, -1))[0, 1])
        except Exception:
            pass

    return {"has_edge": has_edge, "n": len(trades), "n_test": int(test.sum()),
            "win_rate": win_test, "expectancy": exp_test, "naive_exp": bh,
            "prob_current": prob_current, "current_side": float(side[-1])}


def build_playbook(prices: np.ndarray, h: int, costs: Dict[str, float]) -> Dict[str, Any]:
    """Regime + qualifying entry style + illustrative levels for one coin+horizon."""
    s = np.asarray(prices, dtype=float)
    n = len(s)
    if n < 80:
        return {"available": False, "reason": "Not enough history for a playbook."}
    feats = features.compute_features(s)
    feat_mat = feats[_FEAT_COLS].to_numpy(dtype=float)
    close = pd.Series(s)
    atr_pct = _atr_pct(s)

    boll = feats["boll_z"].to_numpy()
    high20 = close.rolling(20).max().shift(1).to_numpy()
    low20 = close.rolling(20).min().shift(1).to_numpy()
    dip_side = np.where(boll < -1.0, 1.0, np.where(boll > 1.0, -1.0, 0.0))
    brk_side = np.where(s > high20, 1.0, np.where(s < low20, -1.0, 0.0))

    cost = 2.0 * (costs["taker_fee"] + costs["slippage"]) + costs["funding_8h"] * (h / 8.0)
    evald = {"Dip": _eval_style(s, dip_side, feat_mat, h, cost, atr_pct),
             "Breakout": _eval_style(s, brk_side, feat_mat, h, cost, atr_pct)}

    regime_val = float(feats["regime"].iloc[-1]) if np.isfinite(feats["regime"].iloc[-1]) else 0.0
    is_trend = regime_val >= REGIME_TREND
    regime = "Trend" if is_trend else "Range"
    fitting = "Breakout" if is_trend else "Dip"
    chosen = evald[fitting]
    other = evald["Dip" if fitting == "Breakout" else "Breakout"]

    base = {"available": True, "regime": regime, "fitting_style": fitting,
            "k_sl": K_SL, "k_tp": K_TP, "prob_threshold": PROB_THRESHOLD,
            "edge_dip": evald["Dip"].get("has_edge", False),
            "edge_breakout": evald["Breakout"].get("has_edge", False)}

    if not chosen.get("has_edge"):
        why = ("the other style had no measured edge either"
               if not other.get("has_edge") else
               f"the {('breakout' if fitting=='Dip' else 'dip')} style showed some edge but is "
               f"against the current {regime.lower()} regime, so it's not surfaced")
        base.update({"style": "No-trade",
                     "reason": f"No measured out-of-sample edge in the regime-fitting "
                               f"{fitting.lower()} style ({why})."})
        return base

    # --- qualifying style: illustrative levels from current price + ATR ---
    cur = float(s[-1])
    atrp = float(atr_pct[-1]) if np.isfinite(atr_pct[-1]) else 0.0
    sma20 = float(close.tail(20).mean())
    std20 = float(close.tail(20).std()) if n >= 20 else 0.0
    boll_now = float(feats["boll_z"].iloc[-1]) if np.isfinite(feats["boll_z"].iloc[-1]) else 0.0
    slope_now = float(feats["sma20_slope"].iloc[-1]) if np.isfinite(feats["sma20_slope"].iloc[-1]) else 0.0

    side = chosen.get("current_side", 0.0)
    if side == 0.0:  # no live trigger -> illustrate the style's natural side
        side = (-1.0 if boll_now > 0 else 1.0) if fitting == "Dip" else (1.0 if slope_now >= 0 else -1.0)
    long = side > 0

    if fitting == "Dip":
        entry_zone = (sma20 - 2 * std20) if long else (sma20 + 2 * std20)
        zone_text = "limit near the lower Bollinger band" if long else "limit near the upper Bollinger band"
    else:
        entry_zone = float(np.nanmax(s[-20:])) if long else float(np.nanmin(s[-20:]))
        zone_text = "stop-entry above the recent 20-bar high" if long else "stop-entry below the recent 20-bar low"

    stop = entry_zone * (1 - side * K_SL * atrp)
    target = entry_zone * (1 + side * K_TP * atrp)
    prob = chosen.get("prob_current")
    skip = bool(prob is not None and prob < PROB_THRESHOLD)

    base.update({
        "style": fitting, "side": "Long" if long else "Short",
        "entry_zone": entry_zone, "entry_text": zone_text,
        "stop": stop, "target": target,
        "sl_pct": K_SL * atrp, "tp_pct": K_TP * atrp, "atr_pct": atrp,
        "win_rate": chosen.get("win_rate"), "expectancy": chosen.get("expectancy"),
        "naive_exp": chosen.get("naive_exp"), "n_test": chosen.get("n_test"),
        "n_setups": chosen.get("n"), "probability": prob, "skip_low_conf": skip,
        "current_price": cur,
    })
    return base


def position_sizing(sl_pct: float, win_rate: Optional[float], prob: Optional[float],
                    capital: float, risk_fraction: float = 0.005) -> Dict[str, float]:
    """Volatility-targeted risk + fractional-Kelly cap (illustrative)."""
    sl_pct = max(sl_pct, 1e-6)
    rupee_risk = capital * risk_fraction
    notional = rupee_risk / sl_pct            # so an SL hit loses ~risk_fraction of capital
    p = prob if prob is not None else (win_rate if win_rate is not None else 0.0)
    b = K_TP / K_SL
    kelly = max(0.0, p - (1.0 - p) / b)
    frac_kelly = min(0.25 * kelly, 0.20)      # quarter-Kelly, capped at 20% of capital
    return {"rupee_risk": rupee_risk, "notional": notional,
            "frac_kelly_pct": frac_kelly * 100.0, "risk_fraction_pct": risk_fraction * 100.0}
