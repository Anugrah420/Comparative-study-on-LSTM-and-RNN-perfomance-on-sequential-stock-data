"""Figures: equity/drawdown (PNG + interactive HTML), gradient decay, fold losses."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots

from metrics import drawdown_series, format_inr

GREY, BH, STRAT, CRASH = "#9e9e9e", "#1f77b4", "#d62728", "#f4a582"


def _rupee_axis(value, _pos=None):
    if value >= 1e7:
        return f"₹{value / 1e7:.1f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.1f} L"
    return format_inr(value, 0)


def _curves(price: pd.Series, bt: dict, capital: float, oos_start: pd.Timestamp):
    """B&H reference (in-sample, rebased to ₹capital at OOS start) + OOS curves."""
    ref = capital * price.loc[:oos_start] / price.loc[oos_start]
    start = pd.Series([capital], index=[oos_start])
    bh = pd.concat([start, bt["bh_equity"]])
    strat = pd.concat([start, bt["strategy_equity"]])
    return ref, bh, strat


def equity_drawdown_figure(price, bt, episodes, labelled, capital, oos_start,
                           model_name, png_path, html_path, log_scale=True):
    ref, bh, strat = _curves(price, bt, capital, oos_start)
    dd_strat, dd_bh = drawdown_series(strat) * 100, drawdown_series(bh) * 100
    end = price.index[-1]

    # ---------------- matplotlib PNG ----------------
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    ax.axvspan(price.index[0], oos_start, color=GREY, alpha=0.25, lw=0)
    ax.text(price.index[0], 0.98, "  In-sample: initial training window\n"
            "  (Buy & Hold shown for reference only;\n  strategy not evaluated here)",
            transform=ax.get_xaxis_transform(), va="top", fontsize=8.5, color="#444")
    for ep in episodes:
        ax.axvspan(ep["peak"], ep["recovery"] or end, color=CRASH, alpha=0.18, lw=0)
    ax.plot(ref.index, ref.values, color=BH, ls="--", lw=1.0, alpha=0.6,
            label="Buy & Hold (in-sample reference)")
    ax.plot(bh.index, bh.values, color=BH, lw=1.5, label="Buy & Hold (out-of-sample)")
    ax.plot(strat.index, strat.values, color=STRAT, lw=1.5, label=f"{model_name} strategy")
    for ep in episodes:
        if ep["label"] in labelled:
            y = ref.reindex([ep["trough"]]).iloc[0] if ep["trough"] <= oos_start \
                else bh.asof(ep["trough"])
            ax.annotate(f"{ep['label']}\nB&H MDD {ep['depth_pct']:.1f}%",
                        xy=(ep["trough"], y), xytext=(0, -38), textcoords="offset points",
                        ha="center", fontsize=8.5, color="#7f2704",
                        arrowprops=dict(arrowstyle="-", color="#7f2704", lw=0.8))
    if log_scale:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_rupee_axis))
    ax.set_ylabel("Portfolio value (₹, log scale)" if log_scale else "Portfolio value (₹)")
    ax.set_title(f"AAPL daily, 20 years — {model_name} strategy vs Buy & Hold "
                 f"({format_inr(capital, 0)} at out-of-sample start)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82), fontsize=9, frameon=False)
    ax.grid(alpha=0.3, which="both")

    ax2.axvspan(price.index[0], oos_start, color=GREY, alpha=0.25, lw=0)
    ax2.fill_between(dd_bh.index, dd_bh.values, 0, color=BH, alpha=0.15, label="Buy & Hold")
    ax2.plot(dd_strat.index, dd_strat.values, color=STRAT, lw=1.2, label=f"{model_name} strategy")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", fontsize=9, frameon=False)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # ---------------- plotly HTML (log/linear toggle) ----------------
    pf = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                       vertical_spacing=0.05)
    pf.add_vrect(x0=price.index[0], x1=oos_start, fillcolor=GREY, opacity=0.25,
                 line_width=0, annotation_text="In-sample (not evaluated)",
                 annotation_position="top left")
    for ep in episodes:
        pf.add_vrect(x0=ep["peak"], x1=ep["recovery"] or end, fillcolor=CRASH,
                     opacity=0.2, line_width=0,
                     annotation_text=(f"{ep['label']}: {ep['depth_pct']:.1f}%"
                                      if ep["label"] in labelled else None),
                     annotation_position="bottom left")
    hover = "%{x|%d %b %Y}<br>%{customdata}<extra>%{fullData.name}</extra>"
    for series, name, style in [(ref, "Buy & Hold (in-sample reference)", dict(color=BH, dash="dash")),
                                (bh, "Buy & Hold (out-of-sample)", dict(color=BH)),
                                (strat, f"{model_name} strategy", dict(color=STRAT))]:
        pf.add_trace(go.Scatter(x=series.index, y=series.values, name=name, line=style,
                                customdata=[format_inr(v, 0) for v in series.values],
                                hovertemplate=hover), row=1, col=1)
    pf.add_trace(go.Scatter(x=dd_bh.index, y=dd_bh.values, name="B&H drawdown %",
                            fill="tozeroy", line=dict(color=BH, width=0.5)), row=2, col=1)
    pf.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat.values, name="Strategy drawdown %",
                            line=dict(color=STRAT)), row=2, col=1)
    pf.update_yaxes(type="log" if log_scale else "linear", title="Portfolio value (₹)", row=1, col=1)
    pf.update_yaxes(title="Drawdown (%)", row=2, col=1)
    pf.update_layout(
        title=f"AAPL 20y — {model_name} vs Buy & Hold", height=780, template="plotly_white",
        updatemenus=[dict(type="buttons", direction="left", x=1.0, y=1.08, xanchor="right",
                          buttons=[dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
                                   dict(label="Linear", method="relayout",
                                        args=[{"yaxis.type": "linear"}])])])
    pf.write_html(html_path, include_plotlyjs="cdn")


def gradient_decay_figure(profiles: dict[int, np.ndarray], input_profiles: dict[int, np.ndarray],
                          model_name, png_path, html_path):
    """||dLoss/dh_t|| (and ||dLoss/dx_t||) vs timestep (1 = oldest), log y."""
    stack = np.stack(list(profiles.values()))           # (folds, layers, window)
    steps = np.arange(1, stack.shape[2] + 1)
    mean_top, mean_bottom = stack[:, -1].mean(0), stack[:, 0].mean(0)
    mean_input = np.stack(list(input_profiles.values())).mean(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fold, prof in profiles.items():
        ax.plot(steps, prof[-1], color=STRAT, alpha=0.15, lw=0.8)
    ax.plot(steps, mean_top, color=STRAT, lw=2.2, label="Top layer, mean over folds")
    ax.plot(steps, mean_bottom, color=BH, lw=1.8, ls="--", label="Bottom layer, mean over folds")
    ax.plot(steps, mean_input, color="#2ca02c", lw=1.8, ls=":",
            label="‖∂Loss/∂xₜ‖ input sensitivity (compare with LSTM)")
    ax.set_yscale("log")
    ax.set_xlabel("Timestep t within the 60-day window (1 = oldest, 60 = most recent)")
    ax.set_ylabel("Mean gradient norm (log scale)")
    ax.set_title(f"{model_name}: gradient flow back through time (faint lines = individual folds)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    pf = go.Figure()
    for fold, prof in profiles.items():
        pf.add_trace(go.Scatter(x=steps, y=prof[-1], name=f"fold {fold} (top)",
                                line=dict(color=STRAT, width=0.8), opacity=0.25))
    pf.add_trace(go.Scatter(x=steps, y=mean_top, name="Top layer mean", line=dict(color=STRAT, width=3)))
    pf.add_trace(go.Scatter(x=steps, y=mean_bottom, name="Bottom layer mean",
                            line=dict(color=BH, width=2, dash="dash")))
    pf.add_trace(go.Scatter(x=steps, y=mean_input, name="‖∂Loss/∂xₜ‖ (inputs)",
                            line=dict(color="#2ca02c", width=2, dash="dot")))
    pf.update_layout(title=f"{model_name}: gradient norm vs timestep", template="plotly_white",
                     xaxis_title="Timestep (1 = oldest)", yaxis_type="log",
                     yaxis_title="Mean gradient norm")
    pf.write_html(html_path, include_plotlyjs="cdn")


def fold_loss_figure(fold_logs: list[dict], model_name, png_path, html_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    pf = go.Figure()
    cmap = plt.get_cmap("viridis", len(fold_logs))
    for i, log in enumerate(fold_logs):
        ep = np.arange(1, len(log["val_loss"]) + 1)
        ax.plot(ep, log["val_loss"], color=cmap(i), lw=1.2, label=f"fold {log['fold']}")
        ax.plot(ep, log["train_loss"], color=cmap(i), lw=0.6, ls=":")
        pf.add_trace(go.Scatter(x=ep, y=log["val_loss"], name=f"fold {log['fold']} val"))
        pf.add_trace(go.Scatter(x=ep, y=log["train_loss"], name=f"fold {log['fold']} train",
                                line=dict(dash="dot"), visible="legendonly"))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled target)")
    ax.set_title(f"{model_name}: validation (solid) and training (dotted) loss per fold")
    ax.legend(ncol=4, fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    pf.update_layout(title=f"{model_name}: loss per fold", template="plotly_white",
                     xaxis_title="Epoch", yaxis_type="log", yaxis_title="MSE (scaled)")
    pf.write_html(html_path, include_plotlyjs="cdn")
