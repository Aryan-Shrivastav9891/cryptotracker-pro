#!/usr/bin/env python3
"""Time-based train/test split for the downloaded crypto dataset.

Time-series data must NOT be shuffled: you train on the earlier period and test
on the most recent period, so the model is only ever evaluated on the "future"
relative to its training data. This script shows:

  1. A chronological train/test split (by fraction, and by an explicit cutoff date).
  2. Building a prediction target without leakage (shift the target back by 1).
  3. Scaling features using TRAIN statistics only (fit on train, transform test).
  4. Walk-forward cross-validation with sklearn's TimeSeriesSplit.

Install:
    pip install pandas numpy scikit-learn

Run (after fetch_crypto_data.py has produced the CSV):
    python scripts/time_split.py
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CSV_PATH = "data/BTC_USDT_1d.csv"   # produced by fetch_crypto_data.py
TEST_SIZE = 0.2                      # last 20% (most recent) -> test
CUTOFF_DATE = "2024-06-01"         # alternative: everything on/after -> test
HORIZON = 1                         # predict this many steps ahead
TASK = "classification"            # "classification" (up/down) or "regression"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV with a proper, sorted, tz-aware datetime index."""
    df = pd.read_csv(path, index_col="datetime", parse_dates=["datetime"])
    df = df.sort_index()
    return df


# --------------------------------------------------------------------------- #
# Target + feature matrix
# --------------------------------------------------------------------------- #
def build_xy(df: pd.DataFrame, horizon: int, task: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features X and a leakage-free target y shifted `horizon` ahead.

    The target uses the FUTURE close, so we shift it back to align each row's
    features with the outcome that follows them. Rows without a known future
    (the last `horizon` rows) and warm-up NaN rows are dropped.
    """
    data = df.copy()

    future_close = data["close"].shift(-horizon)
    if task == "regression":
        # Predict the future log-return over the horizon.
        data["target"] = np.log(future_close / data["close"])
    else:
        # Predict direction: 1 if price goes up over the horizon, else 0.
        data["target"] = (future_close > data["close"]).astype("float")

    # Features: every engineered numeric column except the target and the
    # raw future-tainted nothing (target already excluded). Drop NaNs from
    # indicator warm-up and the unknown-future tail.
    feature_cols: List[str] = [c for c in data.columns if c != "target"]
    data = data.dropna(subset=feature_cols + ["target"])

    X = data[feature_cols]
    y = data["target"]
    return X, y


# --------------------------------------------------------------------------- #
# Chronological splits (NO shuffling)
# --------------------------------------------------------------------------- #
def split_by_fraction(X: pd.DataFrame, y: pd.Series, test_size: float):
    """Earliest (1 - test_size) -> train; most recent test_size -> test."""
    n_test = int(len(X) * test_size)
    n_train = len(X) - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def split_by_date(X: pd.DataFrame, y: pd.Series, cutoff: str):
    """Everything before `cutoff` -> train; on/after -> test."""
    cutoff_ts = pd.Timestamp(cutoff, tz=X.index.tz)
    train_mask = X.index < cutoff_ts
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    return X_train, X_test, y_train, y_test


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit the scaler on TRAIN only, then transform both (no leakage)."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled, scaler


def describe_split(name: str, X_tr, X_te, y_tr, y_te, task: str) -> None:
    print(f"\n=== {name} ===")
    print(f"  train: {len(X_tr):>5} rows  {X_tr.index.min().date()} → {X_tr.index.max().date()}")
    print(f"  test : {len(X_te):>5} rows  {X_te.index.min().date()} → {X_te.index.max().date()}")
    # Sanity: train must end strictly before test begins.
    assert X_tr.index.max() < X_te.index.min(), "LEAKAGE: train overlaps/after test!"
    if task == "classification":
        print(f"  train up-rate: {y_tr.mean():.1%}   test up-rate: {y_te.mean():.1%}")


# --------------------------------------------------------------------------- #
# Walk-forward CV (for hyper-parameter tuning on the training period)
# --------------------------------------------------------------------------- #
def walk_forward_cv(X_train: pd.DataFrame, n_splits: int = 5) -> None:
    """Expanding-window CV that always validates on a later slice than it trains."""
    print(f"\n=== Walk-forward CV (TimeSeriesSplit, {n_splits} folds) ===")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
        tr_dates = X_train.index[tr_idx]
        va_dates = X_train.index[va_idx]
        print(f"  fold {i}: train {len(tr_idx):>5} "
              f"[{tr_dates.min().date()}→{tr_dates.max().date()}]  "
              f"valid {len(va_idx):>4} [{va_dates.min().date()}→{va_dates.max().date()}]")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_dataset(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH} "
          f"({df.index.min().date()} → {df.index.max().date()})")

    X, y = build_xy(df, HORIZON, TASK)
    print(f"Built X={X.shape}, y={y.shape}  (task={TASK}, horizon={HORIZON})")

    # 1) Split by fraction (most recent TEST_SIZE held out).
    X_tr, X_te, y_tr, y_te = split_by_fraction(X, y, TEST_SIZE)
    describe_split(f"Split by fraction (test_size={TEST_SIZE})", X_tr, X_te, y_tr, y_te, TASK)

    # 2) Split by an explicit cutoff date.
    X_tr2, X_te2, y_tr2, y_te2 = split_by_date(X, y, CUTOFF_DATE)
    if len(X_tr2) and len(X_te2):
        describe_split(f"Split by date (cutoff={CUTOFF_DATE})", X_tr2, X_te2, y_tr2, y_te2, TASK)
    else:
        print(f"\n(cutoff {CUTOFF_DATE} falls outside the data range — skipping date split)")

    # 3) Scale using TRAIN stats only (demonstrated on the fraction split).
    X_tr_s, X_te_s, scaler = scale_train_test(X_tr, X_te)
    print(f"\nScaled with StandardScaler fit on TRAIN only "
          f"(train mean≈0: {np.allclose(X_tr_s.mean(), 0, atol=1e-6)}).")

    # 4) Walk-forward CV on the training period (for model selection).
    walk_forward_cv(X_tr, n_splits=5)

    # `X_tr_s, X_te_s, y_tr, y_te` are now ready to feed any sklearn/Keras model.
    print("\nReady: X_tr_s, X_te_s, y_tr, y_te")


if __name__ == "__main__":
    main()
