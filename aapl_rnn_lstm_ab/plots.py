"""Matplotlib plots for a single run. Each function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_dir


def _save(fig, out_dir, name: str) -> Path:
    path = ensure_dir(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_losses(history: dict, out_dir, label: str = "RNN") -> Path:
    """Train vs validation loss per epoch, with the best epoch marked."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    best = history.get("best_epoch")
    if best:
        ax.axvline(best, color="gray", ls="--", label=f"best epoch ({best})")
    ax.set(xlabel="epoch", ylabel="MSE (scaled)", title=f"{label} training curves")
    ax.legend()
    return _save(fig, out_dir, "losses.png")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates, out_dir) -> Path:
    """Predicted vs actual next-day returns over time, plus a scatter of the two."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    a1.plot(dates, y_true, lw=0.7, label="actual", alpha=0.8)
    a1.plot(dates, y_pred, lw=0.7, label="predicted", alpha=0.8)
    a1.set(title="Test: next-day log return", ylabel="log return")
    a1.legend()
    a2.scatter(y_true, y_pred, s=6, alpha=0.4)
    lim = float(max(np.abs(y_true).max(), np.abs(y_pred).max()))
    a2.plot([-lim, lim], [-lim, lim], color="gray", ls="--")
    a2.set(xlabel="actual", ylabel="predicted", title="Predicted vs actual")
    return _save(fig, out_dir, "predictions.png")


def plot_equity(bt: pd.DataFrame, out_dir, label: str = "RNN") -> Path:
    """Strategy equity curve vs buy-and-hold on the test period."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bt.index, bt["equity"], label=f"{label} strategy")
    ax.plot(bt.index, bt["bh_equity"], label="buy & hold", alpha=0.8)
    ax.set(title="Backtest equity (start = 1.0)", ylabel="equity")
    ax.legend()
    return _save(fig, out_dir, "equity.png")