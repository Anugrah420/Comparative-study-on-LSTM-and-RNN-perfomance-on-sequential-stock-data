"""
metrics.py — evaluation matrix, ₹ formatting, JSON I/O, RNN-vs-LSTM table.
Identical to the RNN experiment (same metric names = same JSON schema).

JSON schema (shared with results_intraday_rnn.json):
{
  "experiment": str, "model": "LSTM" | "SimpleRNN", "ticker": "AAPL",
  "config": {...}, "params": {...},
  "metrics": {"Strategy": {METRIC: value}, "Buy & Hold": {...},
              "Persistence": {...}, ["LSTM (param-matched)": {...}]},
  "diagnostics": {...}, "verdict": str
}
Non-finite numbers are stored as null.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import config as C

METRICS = [
    "Total ROI (%)", "Net Profit (₹)",
    "Directional Accuracy (%)", "Profit Factor", "Total Trades Executed",
    "Max Drawdown (%)", "VaR 95% (%)", "VaR 95% (₹)",
    "Sharpe Ratio", "Sortino Ratio",
    "RMSE (USD)", "MAPE (%)",
]
INR_METRICS = {"Net Profit (₹)", "VaR 95% (₹)"}


# ── helpers ──────────────────────────────────────────────────────────────
def safe_div(a, b):
    try:
        if b is None or not np.isfinite(b) or b == 0:
            return float("nan")
        return float(a) / float(b)
    except Exception:
        return float("nan")


def format_inr(x, decimals: int = 2) -> str:
    """₹ with Indian digit grouping: 1234567.8 -> ₹12,34,567.80"""
    if x is None or not np.isfinite(x):
        return "n/a"
    neg, x = x < 0, abs(float(x))
    s = f"{x:.{decimals}f}"
    ip, dp = (s.split(".") + [""])[:2]
    if len(ip) > 3:
        head, last3, groups = ip[:-3], ip[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:]); head = head[:-2]
        if head:
            groups.insert(0, head)
        ip = ",".join(groups + [last3])
    return f"{'-' if neg else ''}₹{ip}{'.' + dp if decimals else ''}"


def directional_accuracy(close_t, actual_next, pred_next) -> float:
    a = np.sign(np.asarray(actual_next) - np.asarray(close_t))
    p = np.sign(np.asarray(pred_next) - np.asarray(close_t))
    m = (a != 0) & (p != 0)              # persistence (p == 0 everywhere) -> NaN
    return float(np.mean(a[m] == p[m]) * 100) if m.any() else float("nan")


def regression_metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return rmse, mape


# ── evaluation ───────────────────────────────────────────────────────────
def evaluate(bt: dict, close_t=None, actual_next=None, pred_next=None,
             capital: float = C.INITIAL_CAPITAL) -> dict:
    eq, r = bt["equity"], bt["returns"].to_numpy(float)
    final = float(eq.iloc[-1])
    curve = np.r_[capital, eq.to_numpy()]
    mdd = float(np.min(curve / np.maximum.accumulate(curve) - 1) * 100)

    rf_bar = (1 + C.RF_ANNUAL) ** (1 / C.ANN_FACTOR) - 1
    ex = r - rf_bar
    ann = math.sqrt(C.ANN_FACTOR)
    sharpe = safe_div(ex.mean(), ex.std(ddof=1)) * ann
    downside = float(np.sqrt(np.mean(np.minimum(ex, 0) ** 2)))
    sortino = safe_div(ex.mean(), downside) * ann

    var_pct = float(np.percentile(r, C.VAR_PCTL) * 100)   # negative = loss

    pnl = bt["trades"]["pnl"].to_numpy(float) if len(bt["trades"]) else np.array([])
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = safe_div(gp, gl) if gl > 0 else (float("inf") if gp > 0 else float("nan"))

    out = {
        "Total ROI (%)": (final / capital - 1) * 100,
        "Net Profit (₹)": final - capital,
        "Directional Accuracy (%)": float("nan"),
        "Profit Factor": pf,
        "Total Trades Executed": int(len(pnl)),
        "Max Drawdown (%)": mdd,
        "VaR 95% (%)": var_pct,
        "VaR 95% (₹)": var_pct / 100 * capital,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "RMSE (USD)": float("nan"),
        "MAPE (%)": float("nan"),
    }
    if pred_next is not None:
        out["Directional Accuracy (%)"] = directional_accuracy(close_t, actual_next, pred_next)
        out["RMSE (USD)"], out["MAPE (%)"] = regression_metrics(actual_next, pred_next)
    return out


def build_matrix(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(METRICS)


def pretty_matrix(df: pd.DataFrame) -> pd.DataFrame:
    fmt = df.copy().astype(object)
    for m in df.index:
        for col in df.columns:
            v = df.loc[m, col]
            if m in INR_METRICS:
                fmt.loc[m, col] = format_inr(v)
            elif m == "Total Trades Executed":
                fmt.loc[m, col] = "n/a" if pd.isna(v) else f"{int(v):,}"
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                fmt.loc[m, col] = "n/a"
            elif isinstance(v, float) and math.isinf(v):
                fmt.loc[m, col] = "∞ (no losing trades)"
            else:
                fmt.loc[m, col] = f"{v:,.4f}" if m in {"Sharpe Ratio", "Sortino Ratio", "Profit Factor"} else f"{v:,.3f}"
    return fmt


# ── JSON I/O ─────────────────────────────────────────────────────────────
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_results(path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, indent=2, ensure_ascii=False, default=str)
    print(f"[save] results -> {path}")


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def compare_with_rnn(lstm_metrics: dict, rnn_json: dict) -> pd.DataFrame | None:
    """Side-by-side RNN vs LSTM table with deltas (LSTM − RNN)."""
    if not rnn_json:
        return None
    rnn = rnn_json.get("metrics", {}).get("Strategy")
    if not rnn:
        print("[compare] RNN JSON found but has no metrics['Strategy'] block.")
        return None
    rows = []
    for m in METRICS:
        a = rnn.get(m); b = lstm_metrics.get(m)
        a = float("nan") if a is None else float(a)
        b = float("nan") if b is None else float(b)
        rows.append({"Metric": m, "SimpleRNN": a, "LSTM": b,
                     "Δ (LSTM − RNN)": b - a if np.isfinite(a) and np.isfinite(b) else float("nan")})
    return pd.DataFrame(rows).set_index("Metric")
