"""
backtest.py — signal rule + bar-by-bar intraday backtest (test set only).
Identical to the RNN experiment.

Rules
  * Decision at the CLOSE of bar t using the predicted close HORIZON bars ahead
    (HORIZON=6 -> 30 minutes by default; see config.py). The position is
    re-evaluated and can flip at EVERY bar using that bar's fresh HORIZON-
    ahead forecast — it is not "hold for HORIZON bars then exit". This keeps
    the backtest mechanics identical to the 1-bar version; it just acts on a
    noisier-but-larger forecast each time.
  * LONG  if pred/close_t - 1 >  round-trip cost (2 x 0.05 %)
    SHORT if pred/close_t - 1 < -round-trip cost, else FLAT.
  * 100 % of equity per trade, fractional shares, compounded returns.
  * COST_PER_SIDE charged on every entry and every exit (a flip = 2 sides).
  * All positions force-closed at the last bar of each session (no overnight).
  * A "trade" = one entry + its exit (round trip).

FX: the asset (SPY) is USD-priced. We compound the strategy's % returns and express
equity in ₹: equity_₹ = INITIAL_CAPITAL x cumulative growth factor
(i.e. USD/INR is assumed constant over the test window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def generate_signals(pred_next: np.ndarray, close_t: np.ndarray,
                     threshold: float = C.ROUND_TRIP_COST) -> np.ndarray:
    exp_ret = np.asarray(pred_next) / np.asarray(close_t) - 1.0
    return np.where(exp_ret > threshold, 1, np.where(exp_ret < -threshold, -1, 0)).astype(int)


def _result(bars, equity, positions, trades, capital):
    eq = pd.Series(equity, index=bars.index, name="equity")
    ret = eq.pct_change()
    ret.iloc[0] = eq.iloc[0] / capital - 1.0
    return {"equity": eq, "returns": ret.fillna(0.0),
            "positions": pd.Series(positions, index=bars.index),
            "trades": pd.DataFrame(trades, columns=["entry_ts", "exit_ts", "side", "ret", "pnl"])}


def run_backtest(bars: pd.DataFrame, signal: pd.Series,
                 cost: float = C.COST_PER_SIDE, capital: float = C.INITIAL_CAPITAL) -> dict:
    """bars: every regular-session bar of the test days (Close, session).
    signal: Series indexed by decision-bar timestamp (bars without a model
    sample — the first WINDOW-1 bars of each session — default to FLAT)."""
    close = bars["Close"].to_numpy(float)
    sess = bars["session"].to_numpy()
    sig = signal.reindex(bars.index).fillna(0).astype(int).to_numpy()
    last_bar = np.r_[sess[1:] != sess[:-1], True]

    n = len(close)
    equity, positions = np.empty(n), np.zeros(n, int)
    eq, pos, entry_eq, entry_ts, trades = float(capital), 0, None, None, []
    for i in range(n):
        if i > 0 and pos != 0:                        # mark-to-market bar i-1 -> i
            eq *= 1.0 + pos * (close[i] / close[i - 1] - 1.0)
        desired = 0 if last_bar[i] else sig[i]        # force flat at session end
        if desired != pos:
            if pos != 0:                              # exit
                eq *= 1.0 - cost
                trades.append([entry_ts, bars.index[i], pos, eq / entry_eq - 1.0, eq - entry_eq])
            if desired != 0:                          # entry
                entry_eq, entry_ts = eq, bars.index[i]
                eq *= 1.0 - cost
            pos = desired
        equity[i], positions[i] = eq, pos
    return _result(bars, equity, positions, trades, capital)


def buy_and_hold(bars: pd.DataFrame, cost: float = C.COST_PER_SIDE,
                 capital: float = C.INITIAL_CAPITAL) -> dict:
    """Buy at the first test-bar close, hold (incl. overnight), sell at the last."""
    close = bars["Close"].to_numpy(float)
    equity = capital * (1 - cost) * close / close[0]
    equity[-1] *= 1 - cost
    trades = [[bars.index[0], bars.index[-1], 1, equity[-1] / capital - 1, equity[-1] - capital]]
    return _result(bars, equity, np.ones(len(close), int), trades, capital)
