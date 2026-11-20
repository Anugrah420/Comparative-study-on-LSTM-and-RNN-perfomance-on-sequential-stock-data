"""Generic performance, risk and fit metrics (shared by all four experiments)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def safe_div(num: float, den: float, default: float = float("nan")) -> float:
    """Division that returns `default` (or +/-inf for a non-zero numerator) on /0."""
    if den == 0 or not np.isfinite(den):
        if num == 0 or not np.isfinite(num):
            return default
        return math.copysign(float("inf"), num)
    return float(num / den)


def format_inr(value: float, decimals: int = 2) -> str:
    """Rupee string with Indian digit grouping, e.g. 1234567.8 -> '₹12,34,567.80'."""
    if value is None or not np.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if decimals else "")


def max_drawdown(equity) -> float:
    """Most negative peak-to-trough decline, as a fraction (e.g. -0.35)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return float("nan")
    return float(np.min(eq / np.maximum.accumulate(eq) - 1.0))


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def sharpe(daily_ret, rf_annual: float = 0.0) -> float:
    r = np.asarray(daily_ret, dtype=float) - rf_annual / TRADING_DAYS
    return safe_div(r.mean() * math.sqrt(TRADING_DAYS), r.std(ddof=1)) if len(r) > 1 else float("nan")


def sortino(daily_ret, rf_annual: float = 0.0) -> float:
    r = np.asarray(daily_ret, dtype=float) - rf_annual / TRADING_DAYS
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    return safe_div(r.mean() * math.sqrt(TRADING_DAYS), downside)


def historical_var(daily_ret, level: float = 0.95) -> float:
    """Historical VaR as the (1-level) percentile of daily returns (a negative number)."""
    r = np.asarray(daily_ret, dtype=float)
    return float(np.percentile(r, (1 - level) * 100)) if len(r) else float("nan")


def equity_metrics(daily_ret: pd.Series, equity: pd.Series, capital: float,
                   rf_annual: float = 0.0) -> dict:
    final = float(equity.iloc[-1])
    var = historical_var(daily_ret)
    return {
        "total_roi_pct": (final / capital - 1.0) * 100,
        "net_profit_inr": final - capital,
        "final_equity_inr": final,
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "var95_pct": var * 100,
        # ₹ VaR is stated on the initial capital so it is comparable across models.
        "var95_inr_on_initial_capital": var * capital,
        "sharpe": sharpe(daily_ret, rf_annual),
        "sortino": sortino(daily_ret, rf_annual),
        "cagr_pct": ((final / capital) ** (TRADING_DAYS / max(len(daily_ret), 1)) - 1) * 100,
    }


def trade_stats(trade_pnl_inr: list[float]) -> dict:
    pnl = np.asarray(trade_pnl_inr, dtype=float)
    gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
    return {
        "total_trades": int(len(pnl)),
        "profit_factor": safe_div(gross_profit, gross_loss),
        "win_rate_pct": safe_div(float((pnl > 0).sum()) * 100, len(pnl)),
        "gross_profit_inr": gross_profit, "gross_loss_inr": gross_loss,
    }


def directional_accuracy(close_t, pred_next, actual_next) -> float:
    """% of days where the predicted move direction matches the realised one.

    Days with no realised move, or a prediction of exactly no change, are
    excluded (so a persistence forecast has no defined accuracy)."""
    pred_move = np.sign(np.asarray(pred_next) - np.asarray(close_t))
    true_move = np.sign(np.asarray(actual_next) - np.asarray(close_t))
    valid = (pred_move != 0) & (true_move != 0)
    return safe_div(float(np.sum(pred_move[valid] == true_move[valid])) * 100, int(valid.sum()))


def fit_metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    return {"rmse_usd": float(np.sqrt(np.mean((p - a) ** 2))),
            "mape_pct": float(np.mean(np.abs((p - a) / a)) * 100)}


def drawdown_episodes(price: pd.Series) -> list[dict]:
    """Peak -> trough -> recovery episodes of a price/equity series, deepest first.

    recovery is the first date the series regains the prior peak (None if it
    has not recovered by the end of the series)."""
    s = price.dropna()
    peak_val, peak_date = s.iloc[0], s.index[0]
    trough_val, trough_date, in_dd = peak_val, peak_date, False
    episodes = []
    for date, value in s.items():
        if value >= peak_val:
            if in_dd:
                episodes.append({"peak": peak_date, "trough": trough_date, "recovery": date,
                                 "depth_pct": (trough_val / peak_val - 1) * 100})
            peak_val, peak_date, trough_val, trough_date, in_dd = value, date, value, date, False
        else:
            in_dd = True
            if value < trough_val:
                trough_val, trough_date = value, date
    if in_dd:
        episodes.append({"peak": peak_date, "trough": trough_date, "recovery": None,
                         "depth_pct": (trough_val / peak_val - 1) * 100})
    return sorted(episodes, key=lambda e: e["depth_pct"])
