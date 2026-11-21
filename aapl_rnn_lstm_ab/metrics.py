"""Regression and trading metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, directional accuracy and correlation on raw next-day returns.

    `rmse_zero` is the RMSE of always predicting 0, a naive baseline to beat.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_pred.std() > 0 and y_true.std() > 0 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "directional_acc": float(np.mean((y_pred > 0) == (y_true > 0))),
        "corr": corr,
        "rmse_zero": float(np.sqrt(np.mean(y_true ** 2))),
    }


def _sharpe(log_ret: pd.Series) -> float:
    """Annualised Sharpe ratio of daily log returns (risk-free rate = 0)."""
    sd = log_ret.std()
    return float(log_ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve (a negative number)."""
    return float((equity / equity.cummax() - 1.0).min())


def trading_metrics(bt: pd.DataFrame) -> dict:
    """Sharpe, max drawdown and total return for the strategy and for buy-and-hold (bh_*)."""
    return {
        "sharpe": _sharpe(bt["strategy_ret"]),
        "max_drawdown": _max_drawdown(bt["equity"]),
        "total_return": float(bt["equity"].iloc[-1] - 1.0),
        "bh_sharpe": _sharpe(bt["bh_ret"]),
        "bh_max_drawdown": _max_drawdown(bt["bh_equity"]),
        "bh_total_return": float(bt["bh_equity"].iloc[-1] - 1.0),
        "n_trades": int((bt["position"].diff().abs().fillna(bt["position"].abs()) > 0).sum()),
    }