"""
plots.py — figures (PNG via matplotlib, interactive HTML via plotly).

Main figure: last 3 full test days — Actual vs LSTM predicted close, VWAP,
optional RNN prediction (dashed), high-volatility shading, overnight gaps
removed; lower panel: Volume bars + mean forget-gate activation per bar.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C


def _runs(mask: np.ndarray):
    """(start, end) integer index pairs of consecutive True runs."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def plot_training_curve(hist: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["train_loss"], label="train")
    ax.plot(hist["epoch"], hist["val_loss"], label="validation")
    ax.set(xlabel="epoch", ylabel="MSE (scaled)", title="LSTM intraday — training curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_equity(curves: dict, path_png, path_html):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = None
    for name, eq in curves.items():
        x = np.arange(len(eq))
        ax.plot(x, eq.to_numpy(), label=name)
    ax.axhline(C.INITIAL_CAPITAL, color="grey", lw=0.8, ls=":")
    ax.set(title="Test-set equity (₹, compounded, fractional shares)", xlabel="test bar #", ylabel="₹")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path_png, dpi=150); plt.close(fig)

    try:
        import plotly.graph_objects as go
        f = go.Figure()
        for name, eq in curves.items():
            f.add_scatter(x=list(range(len(eq))), y=eq.to_numpy(), name=name, mode="lines")
        f.update_layout(title="Test-set equity (₹)", template="plotly_white",
                        xaxis_title="test bar #", yaxis_title="₹")
        f.write_html(path_html)
    except ImportError:
        print("[plots] plotly not installed — skipped equity HTML")


def plot_gate_vs_volume(f_last: np.ndarray, z_last: np.ndarray, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(z_last, f_last, s=8, alpha=0.4)
    ax.axvline(C.SPIKE_Z, color="red", ls="--", lw=1, label=f"spike z={C.SPIKE_Z:g}")
    ax.set(xlabel="time-of-day-adjusted volume z-score", ylabel="mean forget gate (bar t)",
           title="Forget gate vs volume (test set)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_three_day(test_bars: pd.DataFrame, pred: pd.Series, forget: pd.Series,
                   vol_threshold: float, path_png, path_html, rnn_pred: pd.Series | None = None):
    """pred / rnn_pred indexed by TARGET timestamp (bar being predicted);
    forget indexed by decision-bar timestamp."""
    counts = test_bars.groupby("session").size()
    full = [d for d, n in counts.items() if n == C.BARS_PER_DAY]
    days = (full if len(full) >= 3 else list(counts.index))[-3:]
    sub = test_bars[test_bars["session"].isin(days)].copy()
    sub["pred"] = pred.reindex(sub.index)
    sub["forget"] = forget.reindex(sub.index)
    sub["rnn"] = rnn_pred.reindex(sub.index) if rnn_pred is not None else np.nan
    hv = (sub["ret_std_12"] > vol_threshold).to_numpy()
    runs = _runs(hv)

    # ── matplotlib PNG (integer x-axis => overnight gaps removed) ──────
    x = np.arange(len(sub))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    for s, e in runs:
        for ax in (a1, a2):
            ax.axvspan(s - 0.5, e + 0.5, color="orange", alpha=0.15, lw=0)
    a1.plot(x, sub["Close"], color="black", lw=1.4, label="Actual Close")
    a1.plot(x, sub["pred"], color="tab:blue", lw=1.2, label="LSTM predicted")
    a1.plot(x, sub["VWAP"], color="tab:purple", lw=1, ls=":", label="VWAP")
    if sub["rnn"].notna().any():
        a1.plot(x, sub["rnn"], color="tab:red", lw=1, ls="--", label="RNN predicted")
    a1.set_ylabel("USD"); a1.grid(alpha=0.3)
    a1.set_title(f"AAPL 5m — last {len(days)} test days (shaded = high volatility)")
    a1.legend(loc="upper left", fontsize=8)
    a2.bar(x, sub["Volume"], color="grey", alpha=0.6, width=0.9)
    a2.set_ylabel("Volume")
    a2b = a2.twinx()
    a2b.plot(x, sub["forget"], color="tab:green", lw=1.2, label="mean forget gate")
    a2b.set_ylabel("forget gate", color="tab:green")
    sess = sub["session"].to_numpy()
    starts = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    for s in starts[1:]:
        for ax in (a1, a2):
            ax.axvline(s - 0.5, color="k", lw=0.8, alpha=0.6)
    a2.set_xticks(starts + C.BARS_PER_DAY // 2 if len(starts) else [])
    a2.set_xticklabels([str(sess[s]) for s in starts])
    fig.tight_layout(); fig.savefig(path_png, dpi=150); plt.close(fig)

    # ── plotly HTML (rangebreaks remove nights/weekends/holidays) ──────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[plots] plotly not installed — skipped 3-day HTML"); return

    t = sub.index.tz_localize(None)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.05, specs=[[{}], [{"secondary_y": True}]])
    fig.add_scatter(x=t, y=sub["Close"], name="Actual Close", line=dict(color="black", width=2), row=1, col=1)
    fig.add_scatter(x=t, y=sub["pred"], name="LSTM predicted", line=dict(color="royalblue"), row=1, col=1)
    fig.add_scatter(x=t, y=sub["VWAP"], name="VWAP", line=dict(color="purple", dash="dot"), row=1, col=1)
    if sub["rnn"].notna().any():
        fig.add_scatter(x=t, y=sub["rnn"], name="RNN predicted", line=dict(color="crimson", dash="dash"), row=1, col=1)
    fig.add_bar(x=t, y=sub["Volume"], name="Volume", marker_color="grey", opacity=0.6, row=2, col=1)
    fig.add_scatter(x=t, y=sub["forget"], name="mean forget gate", line=dict(color="green"),
                    row=2, col=1, secondary_y=True)
    step = pd.Timedelta(minutes=C.BAR_MINUTES)
    for s, e in runs:
        fig.add_vrect(x0=t[s], x1=t[e] + step, fillcolor="orange", opacity=0.15, line_width=0, row="all", col=1)
    bdays = pd.bdate_range(t.min().normalize(), t.max().normalize())
    have = set(pd.Index(t.normalize()).unique())
    holidays = [d.strftime("%Y-%m-%d") for d in bdays if d not in have]
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                  dict(bounds=[16, 9.5], pattern="hour"),
                                  dict(values=holidays)])
    fig.update_layout(title=f"AAPL 5m — last {len(days)} test days: Actual vs LSTM (orange = high volatility)",
                      template="plotly_white", height=800, hovermode="x unified")
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="forget gate", row=2, col=1, secondary_y=True)
    fig.write_html(path_html)
