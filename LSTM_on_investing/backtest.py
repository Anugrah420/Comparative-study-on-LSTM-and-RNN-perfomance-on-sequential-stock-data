"""Backtest (v5): continuous long-only EXPOSURE w in [0,1] (rest in cash, 0 %).
Decision at close of day t: target exposure W(t) -> held during day t+1 (so it earns Close(t+1)/Close(t)-1).
Rebalance only if |target - current| >= REBAL_BAND. Cost = COST_PER_SIDE x traded notional (|dw|), plus liquidation of any
open exposure on the last day. Equity = INITIAL_CAPITAL x cumulative growth (fractional shares, USD returns on Rs capital)."""
import numpy as np
import pandas as pd
import config as C


def deadband(w_target, band):
    w, cur = np.zeros(len(w_target)), 0.0
    for i, t in enumerate(w_target):
        if abs(t - cur) >= band - 1e-12:
            cur = t
        w[i] = cur
    return w


def simulate(w, asset):
    """Generic simulator used by the strategy AND every benchmark (identical cost model)."""
    w = np.asarray(w, float); prev = np.r_[0.0, w[:-1]]; turn = np.abs(w - prev)
    cost = C.COST_PER_SIDE * turn; cost[-1] += C.COST_PER_SIDE * w[-1]          # liquidate at the end
    ret = w * asset - cost
    eq = C.INITIAL_CAPITAL * np.cumprod(1.0 + ret)
    pnl = np.diff(np.r_[C.INITIAL_CAPITAL, eq])
    return ret, eq, pnl, int((turn > 1e-9).sum()), turn


def drawdown(eq):
    peak = np.maximum.accumulate(np.r_[C.INITIAL_CAPITAL, eq])[1:]
    return eq / peak - 1.0


def run_backtest(oos):
    """Returns (bt, strategy_daily_pnl, strategy_n_trades, buy_hold_daily_pnl)."""
    asset = oos["Actual_Next"].values / oos["Close"].values - 1.0
    wt = np.clip(oos["W"].values, 0.0, 1.0)
    w = deadband(wt, C.REBAL_BAND)
    ret, eq, pnl, ntr, turn = simulate(w, asset)
    bh_ret, bh_eq, bh_pnl, _, _ = simulate(np.ones(len(asset)), asset)
    bt = pd.DataFrame({"position": w, "target_w": wt, "asset_ret": asset, "turnover": turn, "strat_ret": ret, "equity": eq,
                       "bh_ret": bh_ret, "bh_equity": bh_eq}, index=pd.DatetimeIndex(oos["Next_Date"].values, name="date"))
    bt["dd"] = drawdown(eq); bt["bh_dd"] = drawdown(bh_eq)
    return bt, pnl, ntr, bh_pnl


def rule_benchmarks(feat, oos):
    """Simple, non-ML rules the LSTM must beat to justify itself (same costs, same OOS period)."""
    idx = oos.index; asset = oos["Actual_Next"].values / oos["Close"].values - 1.0
    close, sma, sd = (feat[c].reindex(idx).values for c in ("Close", "SMA_200", "Std_20"))
    rules = {"SMA200 filter": deadband((close > sma).astype(float), 0.0),
             f"Vol-target {int(C.VOL_TARGET*100)}%": deadband(np.clip(C.VOL_TARGET / (sd * np.sqrt(C.TRADING_DAYS) + 1e-12), 0, 1), C.REBAL_BAND)}
    out = {}
    for name, w in rules.items():
        ret, eq, pnl, ntr, _ = simulate(w, asset)
        out[name] = dict(ret=ret, equity=eq, pnl=pnl, n_trades=ntr, exposure=float(w.mean()))
    return out
