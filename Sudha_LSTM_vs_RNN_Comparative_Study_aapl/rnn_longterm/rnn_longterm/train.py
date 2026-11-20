"""Training loop, determinism helpers and the vanishing-gradient diagnostic."""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_determinism(seed: int) -> None:
    """Seed every RNG and ask PyTorch for deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(X, y, batch, shuffle, seed):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=gen)


def train_model(model: nn.Module, X_tr, y_tr, X_va, y_va, *, epochs: int, batch: int,
                lr: float, patience: int, clip: float, seed: int,
                device: torch.device, log_prefix: str = "") -> dict:
    """Adam + MSE, gradient clipping, early stopping on validation loss.

    The best-validation weights are restored before returning.
    """
    model.to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_dl = _loader(X_tr, y_tr, batch, shuffle=True, seed=seed)
    val_dl = _loader(X_va, y_va, batch, shuffle=False, seed=seed)
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, wait = float("inf"), copy.deepcopy(model.state_dict()), 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                total += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
        val_loss = total / max(n, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or wait >= patience:
            print(f"{log_prefix} epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")
        if wait >= patience:
            print(f"{log_prefix} early stop at epoch {epoch}; best epoch {best_epoch} "
                  f"(val {best_val:.5f})")
            break

    model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_val_loss=best_val,
                   epochs_run=len(history["val_loss"]))
    return history


@torch.no_grad()
def predict(model: nn.Module, X, device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(xt), batch):
        out.append(model(xt[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def gradient_norm_profile(model, X, y, device, n_batches: int = 8, batch: int = 64,
                          seed: int = 42):
    """Mean per-sample gradient norms for t = 1..WINDOW on sampled *training* batches.

    Returns (hidden, inputs):
      hidden: (num_layers, WINDOW) array of ||dLoss/dh_t|| per layer (the spec's
              diagnostic; uses the manual unroll in models.py);
      inputs: (WINDOW,) array of ||dLoss/dx_t|| - how much the loss depends on
              each day's input. This one is comparable across RNN and LSTM,
              whereas an LSTM carries long memory in its cell state c, not h.
    Index 0 on the time axis is the oldest step in the window (t = 1).
    """
    model.to(device).eval()
    rng = np.random.default_rng(seed)
    loss_fn = nn.MSELoss()
    profiles, input_profiles = [], []
    for _ in range(n_batches):
        idx = rng.choice(len(X), size=min(batch, len(X)), replace=False)
        xb = torch.as_tensor(X[idx], dtype=torch.float32, device=device).requires_grad_(True)
        yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
        model.zero_grad()
        pred, states = model.unrolled_forward(xb)
        # The manual unroll must reproduce the fused implementation exactly.
        with torch.no_grad():
            ref = model(xb)
        assert torch.allclose(pred, ref, atol=1e-4), "manual unroll diverges from nn.RNN"
        loss_fn(pred, yb).backward()
        profiles.append(np.array([[s.grad.norm(dim=1).mean().item() for s in layer]
                                  for layer in states]))
        input_profiles.append(xb.grad.norm(dim=2).mean(dim=0).cpu().numpy())
    model.zero_grad()
    return np.mean(profiles, axis=0), np.mean(input_profiles, axis=0)


def summarise_gradient_profile(profile: np.ndarray, inputs: np.ndarray | None = None) -> dict:
    """Decay statistics for the top layer (the one feeding the head)."""
    top = profile[-1]
    last = top[-1] if top[-1] > 0 else np.nan
    ratio = float(top[0] / last) if np.isfinite(last) else float("nan")
    below = np.where(top / last < 0.01)[0] if np.isfinite(last) else np.array([])
    # Effective memory: how many steps back the gradient stays >= 1% of the last step.
    memory = int(len(top) - 1 - below.max()) if len(below) else int(len(top))
    return {
        "grad_ratio_t1_over_tW": ratio,
        "log10_decay_t1_over_tW": float(np.log10(ratio)) if ratio > 0 else float("nan"),
        "effective_memory_steps_1pct": memory,
        "per_layer_ratio": [float(l[0] / l[-1]) if l[-1] > 0 else float("nan")
                            for l in profile],
        "input_grad_ratio_t1_over_tW": (float(inputs[0] / inputs[-1])
                                        if inputs is not None and inputs[-1] > 0 else float("nan")),
    }
