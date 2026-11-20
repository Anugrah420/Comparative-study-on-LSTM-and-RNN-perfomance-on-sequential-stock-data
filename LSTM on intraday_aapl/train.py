"""
train.py — seeding, training loop with early stopping, prediction.
Identical to the RNN experiment (model-agnostic).
"""
from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as C


def set_seed(seed: int = C.SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(ws, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(ws.X), torch.from_numpy(ws.y))
    g = torch.Generator().manual_seed(C.SEED)
    return DataLoader(ds, batch_size=C.BATCH, shuffle=shuffle, generator=g, drop_last=False)


def train_model(model: nn.Module, train_ws, val_ws, device, tag: str = "LSTM"):
    """Adam + MSE, grad-clip 1.0, early stopping (patience 8) on val loss.
    Returns the model loaded with its best-validation weights + history."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss_fn = nn.MSELoss()
    tr_dl = _loader(train_ws, C.SHUFFLE_TRAIN_BATCHES)
    va_dl = _loader(val_ws, False)

    best_val, best_state, best_epoch, wait, hist = np.inf, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_dl.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += loss_fn(model(xb), yb).item() * len(xb)
        va_loss /= max(len(va_dl.dataset), 1)
        hist.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        improved = va_loss < best_val - 1e-7
        if improved:
            best_val, best_state, best_epoch, wait = va_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
        print(f"[{tag}] epoch {epoch:02d}  train={tr_loss:.5f}  val={va_loss:.5f}"
              f"{'  *' if improved else ''}")
        if wait >= C.PATIENCE:
            print(f"[{tag}] early stop at epoch {epoch} (best epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)
    print(f"[{tag}] best val MSE={best_val:.5f} @ epoch {best_epoch}  ({time.time() - t0:.1f}s)")
    return model, pd.DataFrame(hist)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(torch.as_tensor(X[s:s + batch], device=device)).cpu().numpy()
           for s in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.empty(0, np.float32)


def inverse_target(scalers, y_scaled: np.ndarray) -> np.ndarray:
    return scalers.y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
