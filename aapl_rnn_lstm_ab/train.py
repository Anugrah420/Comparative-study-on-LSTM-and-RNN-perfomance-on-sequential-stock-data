"""Training loop with early stopping, plus a prediction helper."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from utils import ensure_dir, get_device

GRAD_CLIP = 1.0  # recurrent nets can have exploding gradients


def _run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    """One pass over the loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
    return total / max(n, 1)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Config) -> dict:
    """Train with Adam + MSE and early stopping on val loss.

    Restores the best weights into `model` and saves them to <run_dir>/best_model.pt.
    Returns {"train_loss": [...], "val_loss": [...], "best_epoch": int, "best_path": str}.
    """
    device = get_device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0

    for epoch in range(1, cfg.epochs + 1):
        tr = _run_epoch(model, train_loader, loss_fn, device, opt)
        va = _run_epoch(model, val_loader, loss_fn, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"epoch {epoch:3d}/{cfg.epochs} | train {tr:.5f} | val {va:.5f}")

        if va < best_val - 1e-6:
            best_val, best_epoch, bad = va, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_path = ensure_dir(cfg.run_dir) / "best_model.pt"
    torch.save(model.state_dict(), best_path)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["best_path"] = str(best_path)
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    """Return scaled predictions for every sample in the loader, in order, shape (N,)."""
    model.to(device).eval()
    outs = [model(xb.to(device)).cpu().numpy() for xb, _ in loader]
    return np.concatenate(outs) if outs else np.empty(0)