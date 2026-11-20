"""Expanding-window walk-forward: windowing, per-fold scaling and leakage checks.

Timing convention (used consistently with backtest.py):
  * a sample "ending at row e" uses feature rows e-WINDOW+1 .. e (all known at
    the close of day e) and its target is the close of day e+1;
  * the model for a fold is (re)trained at the close of the day *before* the
    fold's first decision day, so the last training target is a close that is
    already known when the first out-of-sample decision is made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data as data_mod
from data import FEATURE_COLS, PRICE_DIFF_COLS, PRICE_LEVEL_COLS, VOLUME_COLS


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def make_windows(feat: pd.DataFrame, ends: np.ndarray, window: int, scaling: str):
    """Build raw (unscaled) samples whose windows end at the positions `ends`.

    scaling="window_relative" (default): price-level inputs become
      log(P_k / Close_e), MACD terms are divided by Close_e, volume is log1p'd,
      and the target is log(Close_{e+1} / Close_e). The prediction is mapped back
      to a next-day close in USD as Close_e * exp(prediction). This keeps every
      out-of-sample value inside the range the scaler saw in training even
      though AAPL's price rose ~100x over 20 years.
    scaling="global": spec-literal raw columns and a raw USD target, scaled by a
      StandardScaler fitted on the training window (out-of-range at test time).
    Returns X (n, window, features), y_raw (n,), anchor close (n,).
    """
    F = feat[FEATURE_COLS].to_numpy(dtype=np.float64)
    close = feat["Close"].to_numpy(dtype=np.float64)
    target = feat["Target"].to_numpy(dtype=np.float64)
    ends = np.asarray(ends, dtype=int)
    assert ends.min() >= window - 1, "window would start before the first row"
    rows = ends[:, None] - window + 1 + np.arange(window)[None, :]
    X = F[rows].copy()
    anchor = close[ends]
    if scaling == "window_relative":
        col = {c: i for i, c in enumerate(FEATURE_COLS)}
        for c in PRICE_LEVEL_COLS:
            X[:, :, col[c]] = np.log(X[:, :, col[c]] / anchor[:, None])
        for c in PRICE_DIFF_COLS:
            X[:, :, col[c]] = X[:, :, col[c]] / anchor[:, None]
        for c in VOLUME_COLS:
            X[:, :, col[c]] = np.log1p(X[:, :, col[c]])
        y_raw = np.log(target[ends] / anchor)
    elif scaling == "global":
        y_raw = target[ends].copy()
    else:
        raise ValueError("scaling must be 'window_relative' or 'global'")
    return X, y_raw, anchor


def to_next_close(pred_raw: np.ndarray, anchor: np.ndarray, scaling: str) -> np.ndarray:
    """Map an inverse-scaled model output back to a next-day close in USD."""
    return anchor * np.exp(pred_raw) if scaling == "window_relative" else pred_raw


def fold_plan(n_rows: int, initial_train: int, step: int) -> list[tuple[int, int]]:
    """[(test_start, test_end), ...] row positions for each expanding fold."""
    folds, start = [], initial_train
    while start < n_rows:
        end = min(start + step, n_rows)
        folds.append((start, end))
        start = end
    return folds


# --------------------------------------------------------------------------- #
# Leakage assertion
# --------------------------------------------------------------------------- #
def assert_no_lookahead(raw: pd.DataFrame, feat: pd.DataFrame, train_last_pos: int,
                        test_start_pos: int, x_scaler: StandardScaler,
                        y_scaler: StandardScaler, n_fit_samples: int, window: int,
                        fit_last_date: pd.Timestamp) -> None:
    """Fail loudly if any indicator or scaler used data beyond the training window.

    1. Indicators/targets: recompute features from raw bars truncated at the first
       out-of-sample decision date. Every training row must be bit-identical to
       the full-history features, so no rolling/EWM value used later data.
       (The last training target is the close of that decision date, which is
       known when the first out-of-sample decision is made.)
    2. Scalers: both were fitted on exactly the training-subset samples, and the
       newest row they saw is older than the first out-of-sample decision date.
    """
    first_test_date = feat.index[test_start_pos]
    train_dates = feat.index[: train_last_pos + 1]
    truncated, _ = data_mod.build_features(raw.loc[:first_test_date])
    assert truncated.index[-1] == train_dates[-1], "truncated features end on wrong date"
    assert truncated.index.equals(train_dates), "training rows differ after truncation"
    cols = FEATURE_COLS + ["Target"]
    assert np.allclose(truncated[cols].to_numpy(), feat.loc[train_dates, cols].to_numpy(),
                       rtol=0, atol=1e-9), "an indicator/target used future data"
    assert x_scaler.n_samples_seen_ == n_fit_samples * window, "x-scaler saw extra rows"
    assert y_scaler.n_samples_seen_ == n_fit_samples, "y-scaler saw extra rows"
    assert fit_last_date < first_test_date, "scaler fitted on out-of-sample dates"


# --------------------------------------------------------------------------- #
# Walk-forward driver
# --------------------------------------------------------------------------- #
def run_walk_forward(raw: pd.DataFrame, feat: pd.DataFrame, hp: dict, fit_fn,
                     max_folds: int | None = None):
    """Run every fold; `fit_fn(fold_id, X_tr, y_tr, X_va, y_va, X_te)` must return
    (scaled test predictions, fold log dict). Returns (OOS predictions frame,
    fold logs)."""
    window, scaling = hp["WINDOW"], hp["SCALING"]
    folds = fold_plan(len(feat), hp["INITIAL_TRAIN_DAYS"], hp["RETRAIN_EVERY"])
    if max_folds:
        folds = folds[:max_folds]
    raw_next = pd.Series(raw.index[1:], index=raw.index[:-1])  # next trading date
    pieces, logs = [], []

    for k, (start, end) in enumerate(folds, 1):
        train_ends = np.arange(window - 1, start)          # samples inside training window
        n_val = max(1, int(round(len(train_ends) * hp["VAL_FRACTION"])))
        fit_ends, val_ends = train_ends[:-n_val], train_ends[-n_val:]
        test_ends = np.arange(start, end)

        X_fit, y_fit, _ = make_windows(feat, fit_ends, window, scaling)
        X_val, y_val, _ = make_windows(feat, val_ends, window, scaling)
        X_te, _, anchor_te = make_windows(feat, test_ends, window, scaling)

        # Scalers are re-fitted on this fold's training subset only.
        n_feat = X_fit.shape[2]
        x_scaler = StandardScaler().fit(X_fit.reshape(-1, n_feat))
        y_scaler = StandardScaler().fit(y_fit.reshape(-1, 1))
        scale_x = lambda X: x_scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape)
        scale_y = lambda y: y_scaler.transform(y.reshape(-1, 1)).ravel()

        assert_no_lookahead(raw, feat, train_last_pos=start - 1, test_start_pos=start,
                            x_scaler=x_scaler, y_scaler=y_scaler,
                            n_fit_samples=len(fit_ends), window=window,
                            fit_last_date=feat.index[fit_ends[-1]])

        print(f"\n[fold {k}/{len(folds)}] train {feat.index[train_ends[0]].date()}.."
              f"{feat.index[train_ends[-1]].date()} (fit {len(fit_ends)}, val {n_val}) | "
              f"test {feat.index[start].date()}..{feat.index[end - 1].date()} ({len(test_ends)})")
        pred_scaled, log = fit_fn(k, scale_x(X_fit), scale_y(y_fit), scale_x(X_val),
                                  scale_y(y_val), scale_x(X_te))
        pred_raw = y_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()

        dates = feat.index[test_ends]
        pieces.append(pd.DataFrame({
            "decision_date": dates,
            "next_date": raw_next.loc[dates].to_numpy(),
            "close_t": anchor_te,
            "actual_next_close": feat["Target"].to_numpy()[test_ends],
            "pred_next_close": to_next_close(pred_raw, anchor_te, scaling),
            "fold": k,
        }))
        log.update(fold=k, train_start=str(feat.index[train_ends[0]].date()),
                   train_end=str(feat.index[train_ends[-1]].date()),
                   test_start=str(dates[0].date()), test_end=str(dates[-1].date()),
                   n_fit=int(len(fit_ends)), n_val=int(n_val), n_test=int(len(test_ends)),
                   test_target_outside_train_range=float(np.mean(
                       (feat["Target"].to_numpy()[test_ends] > y_fit.max())
                       if scaling == "global" else 0.0)))
        logs.append(log)

    oos = pd.concat(pieces, ignore_index=True)
    assert oos["decision_date"].is_monotonic_increasing and oos["decision_date"].is_unique
    return oos, logs
