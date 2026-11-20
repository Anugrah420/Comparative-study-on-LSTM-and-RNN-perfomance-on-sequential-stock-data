"""Seeding, device selection (CPU/GPU) and training loop.
Loss = mean-variance UTILITY of the position  +  AUX_WEIGHT x Huber(next-day return head).
Utility:  -E[w*r] + 0.5*lam*E[(w*r)^2]  with r standardised by training volatility and lam set so that the unconditional
optimal exposure is 1/RISK_KAPPA (=100 % = Buy & Hold). The network therefore moves away from Buy & Hold only when its
features say risk-adjusted expected return has changed."""
import os, random
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # must be set before CUDA init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C


def set_seed(seed=None):
    seed = C.SEED if seed is None else seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    """C.DEVICE: 'auto' -> CUDA if available else CPU; 'cuda' -> CUDA or a clear error; 'cpu' -> CPU."""
    req = str(C.DEVICE).lower()
    if req not in ("auto", "cuda", "cpu"):
        raise ValueError(f"DEVICE must be auto/cuda/cpu, got {C.DEVICE!r}")
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' requested but PyTorch cannot see a GPU. Install a CUDA build of PyTorch "
                           "(https://pytorch.org/get-started/locally) or run with --device cpu / --device auto.")
    dev = torch.device("cuda" if (req == "cuda" or (req == "auto" and torch.cuda.is_available())) else "cpu")
    if not getattr(get_device, "_printed", False):
        print(f"[device] using {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
        get_device._printed = True
    return dev


def _loss(out, y_s, r_n, lam):
    w = torch.sigmoid(out[:, 0]); wr = w * r_n
    util = -wr.mean() + 0.5 * lam * (wr ** 2).mean()
    aux = F.huber_loss(out[:, 1], y_s, delta=1.0) if C.LOSS == "huber" else F.mse_loss(out[:, 1], y_s)
    return util + C.AUX_WEIGHT * aux


def train_model(model, Xtr, ytr, rtr, Xva, yva, rva, device, seed, lam):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
    Xt, yt, rt, Xv, yv, rv = f32(Xtr), f32(ytr), f32(rtr), f32(Xva), f32(yva), f32(rva)
    g = torch.Generator().manual_seed(seed); n = len(Xt)
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, C.BATCH):
            idx = perm[i:i + C.BATCH]
            opt.zero_grad(); loss = _loss(model(Xt[idx]), yt[idx], rt[idx], lam); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP); opt.step()
        model.eval()
        with torch.no_grad():
            vl = _loss(model(Xv), yv, rv, lam).item()
        if vl < best - 1e-9:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model, dict(best_val_loss=best, best_epoch=best_ep, epochs_run=ep)


@torch.no_grad()
def predict(model, X, device, chunk=1024):
    """Returns (n, 2): [target exposure in (0,1), standardised next-day return prediction]."""
    model.eval(); outs = []
    for i in range(0, len(X), chunk):
        o = model(torch.from_numpy(X[i:i + chunk]).to(device))
        outs.append(torch.stack([torch.sigmoid(o[:, 0]), o[:, 1]], 1).cpu().numpy())
    return np.concatenate(outs)
