"""Simple long/short backtest driven by the sign of the predicted next-day return."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def run_backtest(preds: np.ndarray, y_raw: np.ndarray, dates: pd.DatetimeIndex,
                 cfg: Config) -> pd.DataFrame:
    """Backtest raw-return predictions against realised next-day log returns.

    preds : raw predicted next-day log returns (N,)  (already inverse-scaled)
    y_raw : realised next-day log returns (N,)       (dataset.y_raw)
    dates : decision dates (N,)                      (dataset.dates)

    Rule: at the close of day t go long if pred > 0 else short, hold until t+1.
    Cost: cfg.cost_bps charged per unit of position change (first trade from flat).
    Returns a DataFrame indexed by date with columns:
      signal, position, strategy_ret, bh_ret, equity, bh_equity
    (returns are log returns; equity curves start at 1.0).
    """
    preds = np.asarray(preds, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    if not (len(preds) == len(y_raw) == len(dates)):
        raise ValueError("preds, y_raw and dates must have the same length")

    signal = np.where(preds > 0, 1.0, -1.0)
    position = signal  # decided at close t, earns y_raw[t] (return from t to t+1)
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * cfg.cost_bps / 1e4
    strategy_ret = position * y_raw - cost

    bt = pd.DataFrame(
        {"signal": signal, "position": position, "strategy_ret": strategy_ret, "bh_ret": y_raw},
        index=pd.DatetimeIndex(dates),
    )
    bt["equity"] = np.exp(bt["strategy_ret"].cumsum())
    bt["bh_equity"] = np.exp(bt["bh_ret"].cumsum())
    return bt