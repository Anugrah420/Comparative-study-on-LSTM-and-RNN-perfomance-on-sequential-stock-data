"""Long/flat daily backtest with identical rules for all four experiments.

Rules
-----
* Decision at the close of day t using the forecast of Close_{t+1}.
* LONG (100% equity) for day t+1 if  pred_{t+1} > Close_t * (1 + 2*COST_PER_SIDE),
  i.e. the forecast gain must clear the round-trip cost; otherwise FLAT (cash, 0%).
* Costs: COST_PER_SIDE of equity on every entry and every exit. An open
  position is liquidated (exit cost charged) on the final day so that Net Profit
  is realised. A trade = one entry + its exit (consecutive long days = 1 trade).
* Buy & Hold: buys on the first out-of-sample decision day, sells on the last,
  paying the same per-side cost on both legs.
* FX-neutral: equity in ₹ = INITIAL_CAPITAL x cumulative growth of the USD
  position (fractional shares); USD/INR moves are ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import directional_accuracy, equity_metrics, trade_stats


def run_backtest(oos: pd.DataFrame, capital: float, cost_per_side: float,
                 rf_annual: float = 0.0) -> dict:
    close_t = oos["close_t"].to_numpy(float)
    actual = oos["actual_next_close"].to_numpy(float)
    pred = oos["pred_next_close"].to_numpy(float)
    dates = pd.DatetimeIndex(oos["next_date"])     # returns are realised on t+1

    threshold = 2 * cost_per_side
    pos = (pred > close_t * (1 + threshold)).astype(int)
    asset_ret = actual / close_t - 1.0

    prev = np.concatenate([[0], pos[:-1]])
    changed = (pos != prev).astype(float)
    growth = (1 - cost_per_side * changed) * (1 + pos * asset_ret)
    if pos[-1] == 1:                                 # liquidate the open position
        growth[-1] *= (1 - cost_per_side)
    strat_ret = pd.Series(growth - 1.0, index=dates)
    strat_eq = capital * (1 + strat_ret).cumprod()

    bh_growth = 1 + asset_ret
    bh_growth[0] *= (1 - cost_per_side)
    bh_growth[-1] *= (1 - cost_per_side)
    bh_ret = pd.Series(bh_growth - 1.0, index=dates)
    bh_eq = capital * (1 + bh_ret).cumprod()

    # Trade ledger: contiguous runs of pos == 1, P&L in ₹ including both costs.
    eq_before = np.concatenate([[capital], strat_eq.to_numpy()[:-1]])
    trades, i = [], 0
    while i < len(pos):
        if pos[i] == 1:
            j = i
            while j + 1 < len(pos) and pos[j + 1] == 1:
                j += 1
            # exit day j+1 carries the exit cost (or day j itself if liquidated at the end)
            exit_idx = j + 1 if j + 1 < len(pos) else j
            trades.append({"entry_decision": str(oos["decision_date"].iloc[i].date()),
                           "exit_decision": str(oos["decision_date"].iloc[exit_idx].date()),
                           "days_held": int(j - i + 1),
                           "pnl_inr": float(strat_eq.iloc[exit_idx] - eq_before[i])})
            i = j + 1
        else:
            i += 1

    strat = equity_metrics(strat_ret, strat_eq, capital, rf_annual)
    strat.update(trade_stats([t["pnl_inr"] for t in trades]))
    strat["directional_accuracy_pct"] = directional_accuracy(close_t, pred, actual)
    strat["exposure_pct"] = float(pos.mean() * 100)
    strat["total_costs_inr_approx"] = float(np.sum(cost_per_side * changed * eq_before)
                                            + (cost_per_side * strat_eq.iloc[-1]
                                               if pos[-1] == 1 else 0.0))

    bh = equity_metrics(bh_ret, bh_eq, capital, rf_annual)
    bh.update(trade_stats([float(bh_eq.iloc[-1] - capital)]))
    bh["directional_accuracy_pct"] = float("nan")   # B&H makes no forecast
    bh["exposure_pct"] = 100.0

    return {"position": pd.Series(pos, index=dates), "strategy_returns": strat_ret,
            "strategy_equity": strat_eq, "bh_returns": bh_ret, "bh_equity": bh_eq,
            "strategy_metrics": strat, "bh_metrics": bh, "trades": trades,
            "signal_threshold": threshold}
