"""Live paper-trading / online-learning session — prequential, NO real orders.

Educational only. Every ~minute it fetches the latest CLOSED 1-minute candles via
ccxt, stores them to SQLite (dedup, restart-safe), and evaluates an ONLINE model
PREQUENTIALLY (predict-then-update, so there is no leakage):

  for each new bar:  predict P(up) from past-only features  ->  observe outcome
                     ->  score it  ->  THEN update the model with that bar.

Paper trades act on the NEXT bar (1-bar latency, never the signal-bar close) and pay
taker fee + slippage + funding. We report live hit-rate, win-rate, expectancy, net
Sharpe, Brier (calibration) and SKILL vs a cost-aware naive baseline, plus a running
paper equity curve and a drawdown kill-switch. At session end we say plainly whether
it beat the naive baseline (often it won't). NO orders are ever placed.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib import features, history

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

_DB_DIR = "data/live"
WARMUP = 40            # bars needed before the model starts predicting
BACKFILL = 720         # 1m candles to seed the model on session start (~12h)
KILL_DD = 0.10         # paper-equity drawdown kill-switch (10%)
DEFAULT_COSTS = {"taker_fee": 0.0005, "slippage": 0.0005, "funding_8h": 0.0001}
_FEAT_COLS = ["r1", "r2", "r3", "r5", "sma5_gap", "sma10_gap", "sma20_gap",
              "ema12_gap", "macd", "rsi14", "boll_z", "vol10", "vol20",
              "sma10_slope", "sma20_slope", "regime"]


# --------------------------------------------------------------------------- #
# Storage (SQLite — restart-safe)
# --------------------------------------------------------------------------- #
def _db(symbol: str) -> sqlite3.Connection:
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(os.path.join(_DB_DIR, f"{symbol.upper()}_1m.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS candles "
                 "(ts INTEGER PRIMARY KEY, o REAL, h REAL, l REAL, c REAL, v REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v REAL)")
    return conn


def _meta_get(conn, k, default=None):
    row = conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return row[0] if row else default


def _meta_set(conn, k, v):
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, float(v)))
    conn.commit()


def fetch_1m(symbol: str, limit: int = 720) -> List[list]:
    """Latest CLOSED 1-minute candles via ccxt (drops the still-forming last bar)."""
    ex = history._exchange()
    if ex is None:
        return []
    pair = f"{symbol.upper()}/USDT"
    ex.load_markets()
    if pair not in ex.markets:
        return []
    since = ex.milliseconds() - int((limit + 2) * 60_000)
    rows = ex.fetch_ohlcv(pair, timeframe="1m", since=since, limit=min(limit + 2, 1000))
    return rows[:-1] if rows else []  # drop the forming candle


def _store(conn, rows: List[list]) -> int:
    before = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    conn.executemany("INSERT OR IGNORE INTO candles(ts,o,h,l,c,v) VALUES (?,?,?,?,?,?)",
                     [(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in rows])
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] - before


def _load(conn) -> Tuple[np.ndarray, np.ndarray]:
    cur = conn.execute("SELECT ts, c FROM candles ORDER BY ts")
    data = cur.fetchall()
    ts = np.array([d[0] for d in data], dtype=float)
    closes = np.array([d[1] for d in data], dtype=float)
    return ts, closes


# --------------------------------------------------------------------------- #
# Online model (prequential)
# --------------------------------------------------------------------------- #
class _Online:
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42) if _HAS_SK else None
        self.fitted = False

    def warm(self, X, y):
        if self.clf is None or len(set(y.tolist())) < 2:
            return
        self.scaler = StandardScaler().fit(X)
        self.clf.partial_fit(self.scaler.transform(X), y, classes=np.array([0, 1]))
        self.fitted = True

    def p_up(self, x) -> float:
        if not self.fitted:
            return 0.5
        return float(self.clf.predict_proba(self.scaler.transform(x.reshape(1, -1)))[0, 1])

    def update(self, x, y):
        if self.clf is None or self.scaler is None:
            return
        self.clf.partial_fit(self.scaler.transform(x.reshape(1, -1)), [int(y)])
        self.fitted = True


def _cost(costs: Dict[str, float], hold_min: float = 1.0) -> float:
    return 2.0 * (costs["taker_fee"] + costs["slippage"]) + costs["funding_8h"] * (hold_min / 60.0 / 8.0)


def run_prequential(closes: np.ndarray, live_from_idx: int, threshold: float,
                    costs: Dict[str, float]) -> Dict[str, Any]:
    """Predict-then-update over the closes. Bars before live_from_idx only TRAIN the
    model (warm-up); bars at/after it are scored as the live paper session."""
    n = len(closes)
    feats = features.compute_features(closes)
    X = feats[_FEAT_COLS].to_numpy(dtype=float)
    cost = _cost(costs)
    model = _Online()

    # Warm-up training (no scoring) on bars [WARMUP, live_from_idx).
    warm_rows = [i for i in range(WARMUP, max(WARMUP, live_from_idx)) if i + 1 < n and np.all(np.isfinite(X[i]))]
    if warm_rows:
        Xw = X[warm_rows]
        yw = np.array([1 if closes[i + 1] > closes[i] else 0 for i in warm_rows])
        model.warm(Xw, yw)

    pred_p, pred_y, trade_net, trade_side, eq = [], [], [], [], []
    naive_net = []
    equity = 1.0
    start = max(WARMUP, live_from_idx)
    for i in range(start, n - 1):
        if not np.all(np.isfinite(X[i])):
            continue
        p = model.p_up(X[i])
        label = 1 if closes[i + 1] > closes[i] else 0
        pred_p.append(p); pred_y.append(label)
        side = 1 if p > 0.5 + threshold else (-1 if p < 0.5 - threshold else 0)
        if side != 0:
            net = side * (closes[i + 1] / closes[i] - 1.0) - cost   # latency: act next bar
            trade_net.append(net); trade_side.append(side)
            equity *= (1.0 + net); eq.append(equity)
        naive_net.append((closes[i + 1] / closes[i] - 1.0) - cost)   # always-long baseline
        model.update(X[i], label)                                    # online update AFTER scoring

    res = _metrics(np.array(pred_p), np.array(pred_y), np.array(trade_net),
                   np.array(eq), np.array(naive_net), threshold)
    # Forward PAPER signal for the next (not-yet-closed) bar.
    nxt = model.p_up(X[-1]) if np.all(np.isfinite(X[-1])) else 0.5
    res["next_p"] = float(nxt)
    if res.get("killed"):
        res["next_signal"], res["next_reason"] = "NO-TRADE", "Kill-switch: paper drawdown exceeded limit."
    else:
        res["next_signal"] = ("LONG" if nxt > 0.5 + threshold
                              else "SHORT" if nxt < 0.5 - threshold else "NO-TRADE")
    return res


def _metrics(pred_p, pred_y, trade_net, eq, naive_net, threshold) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_pred": int(len(pred_p)), "n_trades": int(len(trade_net)),
                           "threshold": threshold, "equity_curve": [float(x) for x in eq]}
    if len(pred_p):
        out["hit_rate"] = float(np.mean((pred_p > 0.5) == (pred_y == 1)))
        out["brier"] = float(np.mean((pred_p - pred_y) ** 2))
        # reliability (calibration) curve: predicted vs realised in probability bins
        bins = np.clip(((pred_p * 5).astype(int)), 0, 4)
        rel = []
        for b in range(5):
            m = bins == b
            if m.sum() >= 3:
                rel.append((float(np.mean(pred_p[m])), float(np.mean(pred_y[m])), int(m.sum())))
        out["reliability"] = rel
    if len(trade_net):
        out["win_rate"] = float(np.mean(trade_net > 0))
        out["expectancy"] = float(np.mean(trade_net))
        ann = np.sqrt(525600.0)  # 1-min bars per year
        out["sharpe"] = float(np.mean(trade_net) / np.std(trade_net) * ann) if np.std(trade_net) > 1e-12 else 0.0
        peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
        out["max_dd"] = float(np.min(eq / peak - 1.0)) if len(eq) else 0.0
        out["final_equity"] = float(eq[-1]) if len(eq) else 1.0
    naive_exp = float(np.mean(naive_net)) if len(naive_net) else 0.0
    out["naive_exp"] = naive_exp
    if "expectancy" in out:
        out["skill_vs_naive"] = out["expectancy"] - naive_exp
        out["beat_naive"] = bool(out["expectancy"] > 0 and out["expectancy"] > naive_exp and out["n_trades"] >= 10)
        out["killed"] = bool(out.get("max_dd", 0.0) <= -KILL_DD)
    return out


# --------------------------------------------------------------------------- #
# Session API
# --------------------------------------------------------------------------- #
def ensure_session(symbol: str) -> int:
    """Open/seed the DB; backfill history once; return the live-start candle index."""
    conn = _db(symbol)
    have = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    if have < WARMUP + 5:
        _store(conn, fetch_1m(symbol, BACKFILL))
    start_ts = _meta_get(conn, "session_start_ts")
    ts, _ = _load(conn)
    if start_ts is None:
        start_ts = float(ts[-1]) if len(ts) else 0.0
        _meta_set(conn, "session_start_ts", start_ts)
    idx = int(np.searchsorted(ts, start_ts)) if len(ts) else 0
    conn.close()
    return idx


def tick(symbol: str, threshold: float, costs: Dict[str, float]) -> Dict[str, Any]:
    """Fetch latest candles, store, recompute the prequential live session."""
    conn = _db(symbol)
    added = _store(conn, fetch_1m(symbol, 90))
    ts, closes = _load(conn)
    start_ts = _meta_get(conn, "session_start_ts", float(ts[-1]) if len(ts) else 0.0)
    live_from = int(np.searchsorted(ts, start_ts)) if len(ts) else 0
    conn.close()
    if len(closes) < WARMUP + 2:
        return {"candles": int(len(closes)), "new_candles": int(added),
                "live_bars": 0, "live": {"n_pred": 0, "n_trades": 0},
                "trailing": {"n_pred": 0, "n_trades": 0}}
    live_m = run_prequential(closes, live_from, threshold, costs)        # this session only
    trailing = run_prequential(closes, WARMUP, threshold, costs)          # full backfill, for context
    return {"candles": int(len(closes)), "new_candles": int(added),
            "live_bars": int(len(closes) - live_from), "live": live_m, "trailing": trailing}


def reset_session(symbol: str) -> None:
    """Forget the session start (keeps cached candles) so a new session starts now."""
    conn = _db(symbol)
    ts, _ = _load(conn)
    _meta_set(conn, "session_start_ts", float(ts[-1]) if len(ts) else 0.0)
    conn.close()
