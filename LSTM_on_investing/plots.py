"""Figures (PNG via matplotlib; interactive HTML via plotly if installed)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import config as C
from metrics import fmt_inr

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:                                                  # HTML output is optional
    go = None


def _shade(ax, episodes, ypos=0.97):
    for e in episodes:
        ax.axvspan(e["peak"], e["trough"], color="red", alpha=0.12, lw=0)
        ax.text(e["trough"], ypos, f"{e['label']}\nB&H {e['depth']*100:.0f}%", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=7, color="darkred")


def plot_equity(bt, close_full, episodes, oos_start, rnn_eq, png, html, log_y=True, extras=None):
    ctx = close_full / close_full.iloc[0] * C.INITIAL_CAPITAL
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(ctx.index, ctx, color="silver", lw=0.8, ls=":", label="Buy & Hold, full history (context)")
    a1.plot(bt.index, bt["bh_equity"], color="tab:blue", lw=1.3, label="Buy & Hold (OOS)")
    a1.plot(bt.index, bt["equity"], color="tab:green", lw=1.5, label="LSTM strategy (OOS)")
    for nm, eqs in (extras or {}).items():
        a1.plot(bt.index, eqs, lw=0.9, ls="--", label=nm)
    if rnn_eq is not None:
        a1.plot(rnn_eq.index, rnn_eq["equity"], color="tab:orange", lw=1.1, label="Simple RNN strategy (OOS)")
    for a in (a1, a2):
        a.axvspan(close_full.index[0], oos_start, color="grey", alpha=0.25, lw=0)
    a1.text(close_full.index[0], 0.03, "IN-SAMPLE initial training window\n(includes the 2008 crash)", transform=a1.get_xaxis_transform(), fontsize=8)
    _shade(a1, episodes); _shade(a2, episodes, 0.2)
    if log_y: a1.set_yscale("log")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_inr(v))); a1.set_ylabel("Portfolio value (₹)")
    a1.set_title(f"{C.TICKER} - 20y walk-forward: LSTM strategy vs Buy & Hold (₹1,00,000 start; USD returns, no FX)"); a1.legend(loc="upper left"); a1.grid(alpha=.3)
    a2.fill_between(bt.index, bt["bh_dd"] * 100, 0, color="tab:blue", alpha=.3, label="Buy & Hold DD")
    a2.plot(bt.index, bt["dd"] * 100, color="tab:green", lw=1, label="LSTM strategy DD")
    if rnn_eq is not None:
        e = rnn_eq["equity"].values; a2.plot(rnn_eq.index, (e / np.maximum.accumulate(np.r_[C.INITIAL_CAPITAL, e])[1:] - 1) * 100, color="tab:orange", lw=.8, label="RNN DD")
    a2.set_ylabel("Drawdown (%)"); a2.legend(loc="lower left"); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    if go is None: return
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.7, .3], vertical_spacing=.04)
    f.add_trace(go.Scatter(x=ctx.index, y=ctx, name="B&H full history", line=dict(color="silver", dash="dot")), 1, 1)
    f.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], name="Buy & Hold (OOS)"), 1, 1)
    f.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="LSTM strategy"), 1, 1)
    for nm, eqs in (extras or {}).items(): f.add_trace(go.Scatter(x=bt.index, y=eqs, name=nm, line=dict(dash="dash")), 1, 1)
    if rnn_eq is not None: f.add_trace(go.Scatter(x=rnn_eq.index, y=rnn_eq["equity"], name="RNN strategy"), 1, 1)
    f.add_trace(go.Scatter(x=bt.index, y=bt["bh_dd"] * 100, name="B&H DD %", fill="tozeroy"), 2, 1)
    f.add_trace(go.Scatter(x=bt.index, y=bt["dd"] * 100, name="LSTM DD %"), 2, 1)
    for r in (1, 2):
        f.add_vrect(x0=close_full.index[0], x1=oos_start, fillcolor="grey", opacity=.25, line_width=0, row=r, col=1)
        for e in episodes:
            f.add_vrect(x0=e["peak"], x1=e["trough"], fillcolor="red", opacity=.12, line_width=0, row=r, col=1,
                        annotation_text=f"{e['label']} ({e['depth']*100:.0f}%)" if r == 1 else None, annotation_position="top left")
    f.update_yaxes(type="log" if log_y else "linear", row=1, col=1)
    f.update_layout(title=f"{C.TICKER} LSTM walk-forward equity (₹)", height=800, updatemenus=[dict(type="buttons", x=1.0, y=1.12, buttons=[
        dict(label="Log y", method="relayout", args=[{"yaxis.type": "log"}]),
        dict(label="Linear y", method="relayout", args=[{"yaxis.type": "linear"}])])])
    f.write_html(html)


def plot_gates(oos, feat, gate_res, episodes, png, html):
    idx = oos.index
    ratio = (feat["Close"] / feat["SMA_200"] - 1).reindex(idx)
    fm_top, fm_l1 = gate_res["forget_top_series"], gate_res["forget_l1_series"]
    fold_starts = oos.groupby("Fold").apply(lambda d: d.index[0])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, fm_top, color="tab:purple", lw=1, label="mean forget gate (top layer)")
    ax.plot(idx, fm_l1, color="tab:pink", lw=.8, alpha=.7, label="mean forget gate (layer 1)")
    for d in fold_starts: ax.axvline(d, color="k", lw=.4, ls=":")
    _shade(ax, [e for e in episodes if e["trough"] >= idx[0]])
    ax2 = ax.twinx(); ax2.plot(idx, ratio * 100, color="tab:green", lw=.8, alpha=.6, label="Close/SMA200 - 1 (%)"); ax2.set_ylabel("Close/SMA200-1 (%)")
    ax.set_ylabel("forget-gate activation"); ax.set_title("LSTM forget-gate mean vs 200-day trend (dotted = retrain points; units are not comparable across folds)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower left")
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    if go is None: return
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Scatter(x=idx, y=fm_top, name="forget (top)"), secondary_y=False)
    f.add_trace(go.Scatter(x=idx, y=fm_l1, name="forget (layer 1)"), secondary_y=False)
    f.add_trace(go.Scatter(x=idx, y=ratio * 100, name="Close/SMA200-1 %"), secondary_y=True)
    for e in episodes:
        f.add_vrect(x0=e["peak"], x1=e["trough"], fillcolor="red", opacity=.12, line_width=0, annotation_text=e["label"])
    f.update_layout(title="LSTM forget gate vs 200-day trend", height=500); f.write_html(html)


def plot_weights(bt, episodes, png):
    """Exposure over time: does the model de-risk before/in crashes, or just sit in cash at random?"""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.fill_between(bt.index, bt["position"] * 100, 0, step="post", color="tab:green", alpha=.5)
    _shade(ax, [e for e in episodes if e["trough"] >= bt.index[0]])
    ax.set_ylim(0, 105); ax.set_ylabel("Exposure (%)"); ax.set_title(f"{C.TICKER}: LSTM target exposure over time (Buy & Hold = 100%)")
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
