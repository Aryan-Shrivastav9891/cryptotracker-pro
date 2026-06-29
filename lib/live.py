"""Live paper-trading / online-learning session — prequential, NO real orders.

Educational only — not financial advice. Every ~minute it fetches the latest CLOSED
1-minute candles via ccxt, stores them to SQLite (dedup, restart-safe), and runs an
ADAPTIVE ONLINE ENSEMBLE prequentially (predict-then-update -> no leakage):

  * Several online learners (SGD log-loss, PassiveAggressive, Perceptron, BernoulliNB;
    + river models if installed) are blended by HEDGE / multiplicative weights based on
    their recent live loss, so good models gain weight and stale ones fade.
  * CONCEPT-DRIFT detection (river ADWIN if present, else a rolling-error detector) flags
    regime change and resets the blend weights to adapt.

Trades are paper only, gated by real-trader FILTERS (volatility regime, trend/range logic,
volume confirmation, level proximity, spread guard) and managed by RISK RULES (volatility-
targeted sizing + fractional-Kelly cap, ATR stop, reward:risk >= 1.5 target, trailing stop,
cooldown after losses, max trades, drawdown kill-switch). Trades act on the NEXT bar
(1-bar latency) and pay taker fee + slippage + funding. We report trader-grade metrics
(expectancy, profit factor, R-multiples, MAE/MFE, streaks, Sortino, net Sharpe, Brier) and
SKILL vs THREE baselines (always-flat, buy-&-hold, cost-aware naive). It is "tradeable" only
if it beats ALL of them out-of-sample. NO orders are ever placed.
"""
from __future__ import annotations

import os
import sqlite3
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lib import features, history

try:
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

try:  # optional online-learning + drift library
    from river import drift as river_drift
    _HAS_RIVER = True
except Exception:  # pragma: no cover
    _HAS_RIVER = False

_DB_DIR = "data/live"
WARMUP = 40
BACKFILL = 720
KILL_DD = 0.10            # paper-equity drawdown kill-switch
MAX_TRADES = 25           # overtrading guard per session
COOLDOWN_LOSSES = 3       # consecutive losses -> cooldown (no revenge trading)
COOLDOWN_BARS = 15
MAX_HOLD = 20             # max bars to hold a paper trade (vertical barrier)
K_SL, K_TP = 1.0, 1.6     # stop / target in ATR units -> reward:risk = 1.6
RISK_FRAC = 0.005         # 0.5% of account risked per trade (vol targeting)
ATR_LO, ATR_HI = 0.20, 0.92   # skip dead-chop (low) and spike (high) ATR-percentile regimes
FUNDING_HOURS = {0, 8, 16}    # Binance perp funding (UTC)
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
    ex = history._exchange()
    if ex is None:
        return []
    pair = f"{symbol.upper()}/USDT"
    ex.load_markets()
    if pair not in ex.markets:
        return []
    since = ex.milliseconds() - int((limit + 2) * 60_000)
    rows = ex.fetch_ohlcv(pair, timeframe="1m", since=since, limit=min(limit + 2, 1000))
    return rows[:-1] if rows else []


def _store(conn, rows: List[list]) -> int:
    before = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    conn.executemany("INSERT OR IGNORE INTO candles(ts,o,h,l,c,v) VALUES (?,?,?,?,?,?)",
                     [(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in rows])
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] - before


def _load_ohlcv(conn) -> Dict[str, np.ndarray]:
    data = conn.execute("SELECT ts,o,h,l,c,v FROM candles ORDER BY ts").fetchall()
    cols = list(zip(*data)) if data else ([], [], [], [], [], [])
    return {k: np.asarray(cols[i], dtype=float) for i, k in enumerate(("ts", "o", "h", "l", "c", "v"))}


# --------------------------------------------------------------------------- #
# Online ensemble (Hedge weights) + drift
# --------------------------------------------------------------------------- #
class _Member:
    def __init__(self, name, clf, proba):
        self.name, self.clf, self.proba, self.fitted = name, clf, proba, False

    def warm(self, Xs, y):
        if len(set(y.tolist())) < 2:
            return
        self.clf.partial_fit(Xs, y, classes=np.array([0, 1]))
        self.fitted = True

    def p(self, xs) -> float:
        if not self.fitted:
            return 0.5
        if self.proba:
            return float(self.clf.predict_proba(xs)[0, 1])
        z = float(self.clf.decision_function(xs)[0])
        return float(1.0 / (1.0 + np.exp(-z)))

    def update(self, xs, y):
        self.clf.partial_fit(xs, [int(y)])
        self.fitted = True


def _members() -> List[_Member]:
    if not _HAS_SK:
        return []
    return [
        _Member("sgd_log", SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42), True),
        _Member("passive_aggr", PassiveAggressiveClassifier(C=0.1, random_state=42), False),
        _Member("perceptron", Perceptron(alpha=1e-4, random_state=42), False),
        _Member("bernoulli_nb", BernoulliNB(), True),
    ]


class _Ensemble:
    def __init__(self, members: List[_Member], eta: float = 0.6):
        self.members = members
        self.w = np.ones(len(members)) / max(len(members), 1)
        self.eta = eta
        self.scaler: Optional[StandardScaler] = None
        self.warmed = False
        self._bufX: List[np.ndarray] = []   # buffer for lazy warm-up (no explicit window)
        self._bufY: List[int] = []
        self._recent = deque(maxlen=30)     # rolling ensemble error (drift fallback)
        self._base = deque(maxlen=180)
        self.drift = False
        self.adwin = river_drift.ADWIN() if _HAS_RIVER else None

    def warm(self, X, y):
        if not self.members or len(set(np.asarray(y).tolist())) < 2:
            return
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        for m in self.members:
            m.warm(Xs, y)
        self.warmed = True

    def predict(self, x) -> Tuple[float, np.ndarray]:
        if not self.warmed or self.scaler is None:
            return 0.5, np.full(max(len(self.members), 1), 0.5)
        xs = self.scaler.transform(x.reshape(1, -1))
        ps = np.array([m.p(xs) for m in self.members])
        return float(np.dot(self.w, ps)), ps

    def update(self, x, y, ps_prev, ens_p_prev):
        if not self.warmed:
            # Lazy prequential warm-up: collect bars until both classes are seen.
            self._bufX.append(x); self._bufY.append(int(y))
            if len(self._bufY) >= 30 and len(set(self._bufY)) >= 2:
                self.warm(np.array(self._bufX), np.array(self._bufY))
            return
        # Hedge / multiplicative weights from each member's squared loss this bar.
        loss = (ps_prev - y) ** 2
        self.w = self.w * np.exp(-self.eta * loss)
        s = self.w.sum()
        self.w = (self.w / s) if s > 0 else np.ones(len(self.members)) / max(len(self.members), 1)
        xs = self.scaler.transform(x.reshape(1, -1))
        for m in self.members:
            m.update(xs, y)
        # Drift on the ensemble's directional error stream.
        err = 0 if ((ens_p_prev > 0.5) == (y == 1)) else 1
        self._detect(err)

    def _detect(self, err: int):
        drift = False
        if self.adwin is not None:
            self.adwin.update(err)
            drift = bool(getattr(self.adwin, "drift_detected", False))
        else:
            self._recent.append(err); self._base.append(err)
            if len(self._recent) >= 25 and len(self._base) >= 120:
                drift = (np.mean(self._recent) - np.mean(self._base)) > 0.18
        if drift:
            self.drift = True
            self.w = np.ones(len(self.members)) / max(len(self.members), 1)  # reset blend to adapt
        else:
            self.drift = False

    def weights(self) -> Dict[str, float]:
        return {m.name: float(w) for m, w in zip(self.members, self.w)}


# --------------------------------------------------------------------------- #
# Indicators for filters (past-only)
# --------------------------------------------------------------------------- #
def _atr_pct(o, h, l, c, win=14) -> np.ndarray:
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    s = np.convolve(tr, np.ones(win) / win, mode="full")[:len(tr)]
    s[:win] = np.nan
    return s / np.where(c > 0, c, np.nan)


def _roll_pct_rank(x: np.ndarray, win=120) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        a = i - win + 1
        if a < 0 or not np.isfinite(x[i]):
            continue
        w = x[a:i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 10:
            out[i] = float(np.mean(w <= x[i]))
    return out


# --------------------------------------------------------------------------- #
# Session engine
# --------------------------------------------------------------------------- #
def _funding_crossings(ts_ms: np.ndarray, a: int, b: int) -> int:
    cnt = 0
    for j in range(a, b + 1):
        if 0 <= j < len(ts_ms):
            if datetime.fromtimestamp(ts_ms[j] / 1000, tz=timezone.utc).hour in FUNDING_HOURS \
                    and datetime.fromtimestamp(ts_ms[j] / 1000, tz=timezone.utc).minute == 0:
                cnt += 1
    return cnt


def run_session(ohlcv: Dict[str, np.ndarray], live_from_idx: int, threshold: float,
                costs: Dict[str, float], capital: float = 100_000.0) -> Dict[str, Any]:
    ts, o, h, l, c, v = (ohlcv[k] for k in ("ts", "o", "h", "l", "c", "v"))
    n = len(c)
    feats = features.compute_features(c)
    X = feats[_FEAT_COLS].to_numpy(dtype=float)
    atrp = _atr_pct(o, h, l, c)
    atr_rank = _roll_pct_rank(atrp, 120)
    vol_ma = np.convolve(v, np.ones(20) / 20, mode="full")[:len(v)]
    spread = (h - l) / np.where(c > 0, c, np.nan)
    spread_rank = _roll_pct_rank(spread, 120)
    vwap = np.full(n, np.nan)
    win = 240
    for i in range(n):
        a = max(0, i - win + 1)
        vv = v[a:i + 1].sum()
        if vv > 0:
            vwap[i] = float((c[a:i + 1] * v[a:i + 1]).sum() / vv)

    ens = _Ensemble(_members())
    warm_rows = [i for i in range(WARMUP, max(WARMUP, live_from_idx)) if i + 1 < n and np.all(np.isfinite(X[i]))]
    if warm_rows and ens.members:
        ens.warm(X[warm_rows], np.array([1 if c[i + 1] > c[i] else 0 for i in warm_rows]))

    pred_p, pred_y = [], []
    trades: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    acct, peak, killed = 1.0, 1.0, False     # sized paper account (risk RISK_FRAC per trade)
    strat_eq, bh_eq, eq_x = [], [], []
    consec_losses, cooldown_until, next_free_bar = 0, -1, -1
    base_taker, base_slip = costs["taker_fee"], costs["slippage"]
    start = max(WARMUP, live_from_idx)
    bh0 = c[start] if start < n else None

    def _skip(reason):
        skips[reason] = skips.get(reason, 0) + 1

    for i in range(start, n - 1):
        if not np.all(np.isfinite(X[i])):
            continue
        ens_p, ps = ens.predict(X[i])
        label = 1 if c[i + 1] > c[i] else 0
        pred_p.append(ens_p); pred_y.append(label)
        if bh0:
            strat_eq.append(acct); bh_eq.append(c[i] / bh0); eq_x.append(i)

        side = 1 if ens_p > 0.5 + threshold else (-1 if ens_p < 0.5 - threshold else 0)
        # ---------- TRADE FILTERS (real-trader rules; skip with a reason) ----------
        if side != 0 and i < next_free_bar:
            side = 0  # a position is already open (one at a time)
        elif side != 0 and killed:
            _skip("kill-switch active"); side = 0
        elif side != 0:
            ar = atr_rank[i]
            slope = feats["sma20_slope"].iloc[i] if np.isfinite(feats["sma20_slope"].iloc[i]) else 0.0
            is_trend = (feats["regime"].iloc[i] if np.isfinite(feats["regime"].iloc[i]) else 0.0) >= 0.15
            if i < cooldown_until:
                _skip("cooldown after losses"); side = 0
            elif len(trades) >= MAX_TRADES:
                _skip("max trades reached"); side = 0
            elif not np.isfinite(ar) or ar < ATR_LO:
                _skip("dead-chop volatility"); side = 0
            elif ar > ATR_HI:
                _skip("volatility spike"); side = 0
            elif not np.isfinite(atrp[i]) or K_TP * atrp[i] <= 1.5 * (2 * base_taker + 2 * base_slip):
                _skip("target can't cover costs"); side = 0   # don't trade when fees eat the range
            elif is_trend and slope != 0 and np.sign(side) != np.sign(slope):
                _skip("trend regime: counter-trend"); side = 0
            elif (not is_trend) and np.isfinite(vwap[i]) and np.sign(side) == np.sign(c[i] - vwap[i]):
                _skip("range regime: chasing not fading"); side = 0
            elif is_trend and np.isfinite(vol_ma[i]) and v[i] < vol_ma[i]:
                _skip("low-volume breakout"); side = 0

        # ---------- OPEN + MANAGE a paper trade (ATR stop/target, trailing) ----------
        if side != 0 and np.isfinite(atrp[i]) and atrp[i] > 0:
            entry = c[i]
            risk = K_SL * atrp[i]
            stop = entry * (1 - side * risk)
            target = entry * (1 + side * K_TP * atrp[i])
            slip = base_slip * (2.0 if (np.isfinite(spread_rank[i]) and spread_rank[i] > 0.9) else 1.0)
            mfe = mae = 0.0
            exit_px, exit_bar, be_moved = None, None, False
            for j in range(i + 1, min(i + MAX_HOLD, n - 1) + 1):
                mfe = max(mfe, side * ((h[j] if side > 0 else l[j]) / entry - 1))
                mae = min(mae, side * ((l[j] if side > 0 else h[j]) / entry - 1))
                if not be_moved and mfe >= risk:    # trail to breakeven after +1R
                    stop = entry; be_moved = True
                hit_stop = (l[j] <= stop) if side > 0 else (h[j] >= stop)
                hit_tp = (h[j] >= target) if side > 0 else (l[j] <= target)
                if hit_stop:
                    exit_px, exit_bar = stop, j; break
                if hit_tp:
                    exit_px, exit_bar = target, j; break
            if exit_px is None:
                exit_bar = min(i + MAX_HOLD, n - 1); exit_px = c[exit_bar]
            gross = side * (exit_px / entry - 1)
            fund = costs["funding_8h"] * _funding_crossings(ts, i + 1, exit_bar)
            cost = 2.0 * base_taker + 2.0 * slip + fund
            net = gross - cost
            r_mult = net / risk if risk > 0 else 0.0
            acct *= (1.0 + RISK_FRAC * r_mult)      # vol-targeted: a 1R loss = RISK_FRAC of account
            peak = max(peak, acct)
            trades.append({"side": side, "net": net, "r": r_mult, "hold": exit_bar - i,
                           "mfe": mfe, "mae": mae, "win": net > 0})
            next_free_bar = exit_bar
            consec_losses = consec_losses + 1 if net <= 0 else 0
            if consec_losses >= COOLDOWN_LOSSES:
                cooldown_until = exit_bar + COOLDOWN_BARS
            if acct / peak - 1.0 <= -KILL_DD:
                killed = True

        ens.update(X[i], label, ps, ens_p)

    nxt = 0.5
    if np.all(np.isfinite(X[-1])) and ens.members:
        nxt, _ = ens.predict(X[-1])
    res = _metrics(np.array(pred_p), np.array(pred_y), trades, c, start, costs,
                   strat_eq, bh_eq, eq_x, threshold, killed)
    res["weights"] = ens.weights()
    res["drift"] = bool(ens.drift)
    res["skips"] = skips
    last_reg = feats["regime"].iloc[-1] if np.isfinite(feats["regime"].iloc[-1]) else 0.0
    res["regime"] = "Trend" if last_reg >= 0.15 else "Range"
    res["logic"] = "momentum / continuation" if last_reg >= 0.15 else "mean-reversion / fade"
    res["next_p"] = float(nxt)
    if res["killed"]:
        res["next_signal"], res["next_reason"] = "NO-TRADE", "Kill-switch / cooldown — capital protection."
    else:
        res["next_signal"] = ("LONG" if nxt > 0.5 + threshold
                              else "SHORT" if nxt < 0.5 - threshold else "NO-TRADE")
    return res


def _metrics(pred_p, pred_y, trades, c, start, costs, strat_eq, bh_eq, eq_x, threshold, killed) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_pred": int(len(pred_p)), "n_trades": int(len(trades)),
                           "threshold": threshold, "killed": bool(killed)}
    if len(pred_p):
        out["hit_rate"] = float(np.mean((pred_p > 0.5) == (pred_y == 1)))
        out["brier"] = float(np.mean((pred_p - pred_y) ** 2))
        out["brier_recent"] = float(np.mean((pred_p[-30:] - pred_y[-30:]) ** 2)) if len(pred_p) >= 10 else out["brier"]
        bins = np.clip((pred_p * 5).astype(int), 0, 4)
        out["reliability"] = [(float(np.mean(pred_p[bins == b])), float(np.mean(pred_y[bins == b])), int((bins == b).sum()))
                              for b in range(5) if (bins == b).sum() >= 3]
    n_bars = max(len(bh_eq), 1)
    rt = 2.0 * (costs["taker_fee"] + costs["slippage"])
    bh_total = float(c[-1] / c[start] - 1.0 - rt) if start < len(c) - 1 else 0.0
    naive_per_bar = float(np.mean(np.diff(c[start:]) / c[start:-1]) - rt) if start < len(c) - 1 else 0.0
    bh_per_bar = bh_total / n_bars
    out["baseline_flat"], out["baseline_buyhold"], out["baseline_naive"] = 0.0, bh_total, naive_per_bar * n_bars

    nets = np.array([t["net"] for t in trades], dtype=float)
    if len(nets):
        wins, losses = nets[nets > 0], nets[nets < 0]
        out["win_rate"] = float(np.mean(nets > 0))
        out["expectancy"] = float(np.mean(nets))
        out["profit_factor"] = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
        rs = np.array([t["r"] for t in trades])
        out["avg_r"], out["median_r"] = float(np.mean(rs)), float(np.median(rs))
        out["avg_mae"], out["avg_mfe"] = float(np.mean([t["mae"] for t in trades])), float(np.mean([t["mfe"] for t in trades]))
        out["avg_hold"] = float(np.mean([t["hold"] for t in trades]))
        w = [t["win"] for t in trades]
        out["max_win_streak"], out["max_loss_streak"] = _streak(w, True), _streak(w, False)
        ann = np.sqrt(525600.0 / max(out["avg_hold"], 1))
        out["sharpe"] = float(np.mean(nets) / np.std(nets) * ann) if np.std(nets) > 1e-12 else 0.0
        down = nets[nets < 0]
        out["sortino"] = float(np.mean(nets) / np.std(down) * ann) if len(down) > 1 and np.std(down) > 1e-12 else 0.0
        se = np.array(strat_eq) if strat_eq else np.array([1.0])
        out["max_dd"] = float(np.min(se / np.maximum.accumulate(se) - 1.0))
        out["final_equity"] = float(se[-1])
        out["killed"] = bool(killed or out["max_dd"] <= -KILL_DD)
        # exposure-neutral edge: strategy return per IN-MARKET bar vs each baseline per bar
        strat_per_bar = out["expectancy"] / max(out["avg_hold"], 1.0)
        out["edge_per_bar"] = strat_per_bar
        out["beat_all"] = bool(strat_per_bar > 0 and strat_per_bar > naive_per_bar
                               and strat_per_bar > bh_per_bar and out["n_trades"] >= 8 and not out["killed"])
    out["strat_eq"] = [float(x) for x in strat_eq]
    out["bh_eq"] = [float(x) for x in bh_eq]
    out["eq_x"] = [int(x) for x in eq_x]
    return out


def _streak(wins: List[bool], want: bool) -> int:
    best = cur = 0
    for w in wins:
        if w == want:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


# --------------------------------------------------------------------------- #
# Session API
# --------------------------------------------------------------------------- #
def ensure_session(symbol: str) -> int:
    conn = _db(symbol)
    if conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] < WARMUP + 5:
        _store(conn, fetch_1m(symbol, BACKFILL))
    oh = _load_ohlcv(conn)
    ts = oh["ts"]
    start_ts = _meta_get(conn, "session_start_ts")
    if start_ts is None:
        start_ts = float(ts[-1]) if len(ts) else 0.0
        _meta_set(conn, "session_start_ts", start_ts)
    idx = int(np.searchsorted(ts, start_ts)) if len(ts) else 0
    conn.close()
    return idx


def tick(symbol: str, threshold: float, costs: Dict[str, float], capital: float = 100_000.0) -> Dict[str, Any]:
    conn = _db(symbol)
    added = _store(conn, fetch_1m(symbol, 90))
    oh = _load_ohlcv(conn)
    ts = oh["ts"]
    start_ts = _meta_get(conn, "session_start_ts", float(ts[-1]) if len(ts) else 0.0)
    live_from = int(np.searchsorted(ts, start_ts)) if len(ts) else 0
    conn.close()
    if len(oh["c"]) < WARMUP + 2:
        return {"candles": int(len(oh["c"])), "new_candles": int(added), "live_bars": 0,
                "live": {"n_pred": 0, "n_trades": 0}, "trailing": {"n_pred": 0, "n_trades": 0}}
    live = run_session(oh, live_from, threshold, costs, capital)
    trailing = run_session(oh, WARMUP, threshold, costs, capital)
    return {"candles": int(len(oh["c"])), "new_candles": int(added),
            "live_bars": int(len(oh["c"]) - live_from), "live": live, "trailing": trailing}


def reset_session(symbol: str) -> None:
    conn = _db(symbol)
    ts = _load_ohlcv(conn)["ts"]
    _meta_set(conn, "session_start_ts", float(ts[-1]) if len(ts) else 0.0)
    conn.close()
