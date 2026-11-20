"""
models.py — recurrent regressors + LSTM gate inspection.

SimpleRNNRegressor is kept here (identical to the RNN experiment) so both
experiments import from the same file; this experiment trains LSTMRegressor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config as C


class LSTMRegressor(nn.Module):
    """nn.LSTM (batch_first) -> Linear head on the LAST timestep."""

    def __init__(self, n_features: int, hidden: int = C.HIDDEN,
                 num_layers: int = C.NUM_LAYERS, dropout: float = C.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: [B, T, F]
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class SimpleRNNRegressor(nn.Module):
    """Counterpart model (same hyper-parameters, tanh RNN cell)."""

    def __init__(self, n_features: int, hidden: int = C.HIDDEN,
                 num_layers: int = C.NUM_LAYERS, dropout: float = C.DROPOUT):
        super().__init__()
        self.rnn = nn.RNN(n_features, hidden, num_layers, batch_first=True,
                          nonlinearity="tanh", dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


# ── Parameter accounting ─────────────────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return {"total": total, "recurrent": total - head, "head": head}


def param_matched_hidden(n_features: int, target_hidden: int = C.HIDDEN,
                         num_layers: int = C.NUM_LAYERS) -> int:
    """Hidden size h for which an LSTM has ~ the same recurrent parameter
    count as the Simple RNN with `target_hidden` units."""
    rnn_p = count_params(SimpleRNNRegressor(n_features, target_hidden, num_layers))["recurrent"]
    best = min(range(4, target_hidden + 1),
               key=lambda h: abs(count_params(LSTMRegressor(n_features, h, num_layers))["recurrent"] - rnn_p))
    return best


def print_param_report(lstm: LSTMRegressor, n_features: int):
    lp = count_params(lstm)
    rp = count_params(SimpleRNNRegressor(n_features))
    ratio = lp["recurrent"] / rp["recurrent"]
    print(f"[model] LSTM params: total={lp['total']:,} recurrent={lp['recurrent']:,} head={lp['head']:,}")
    print(f"[model] Simple RNN (same config) recurrent params: {rp['recurrent']:,}")
    print(f"[model] ⚠ CONFOUND: LSTM has {ratio:.2f}x the recurrent parameters of the RNN "
          "(4 gates vs 1). Any LSTM advantage may come from capacity, not gating. "
          "Run with --param-matched for a capacity-controlled robustness check.")
    return {"lstm": lp, "rnn_reference": rp, "recurrent_ratio": ratio}


# ── Gate inspection ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_gates(model: LSTMRegressor, X: np.ndarray, device, batch: int = 256) -> dict:
    """Manually unroll the trained nn.LSTM (an LSTMCell-equivalent loop that
    re-uses the trained weights) to expose per-timestep gate activations.

    PyTorch gate order in weight_ih/weight_hh is (input, forget, cell, output).
    Returns forget/input gates shaped [N, layers, T, hidden] (float16) and the
    head prediction, which is asserted to match model(X).
    """
    model.eval()
    lstm = model.lstm
    L, H = lstm.num_layers, lstm.hidden_size
    f_all, i_all, preds = [], [], []

    for s in range(0, len(X), batch):
        x = torch.as_tensor(X[s:s + batch], device=device)
        B, T, _ = x.shape
        inp = x
        f_layers, i_layers = [], []
        for k in range(L):
            W_ih = getattr(lstm, f"weight_ih_l{k}")
            W_hh = getattr(lstm, f"weight_hh_l{k}")
            b = getattr(lstm, f"bias_ih_l{k}") + getattr(lstm, f"bias_hh_l{k}")
            x_proj = inp @ W_ih.T + b                          # [B, T, 4H]
            h = torch.zeros(B, H, device=device)
            c = torch.zeros(B, H, device=device)
            hs, fs, is_ = [], [], []
            for t in range(T):
                g = x_proj[:, t] + h @ W_hh.T
                i_g, f_g, c_g, o_g = g.chunk(4, dim=-1)
                i_g, f_g, o_g = torch.sigmoid(i_g), torch.sigmoid(f_g), torch.sigmoid(o_g)
                c = f_g * c + i_g * torch.tanh(c_g)
                h = o_g * torch.tanh(c)
                hs.append(h); fs.append(f_g); is_.append(i_g)
            inp = torch.stack(hs, dim=1)                       # eval: no inter-layer dropout
            f_layers.append(torch.stack(fs, dim=1))
            i_layers.append(torch.stack(is_, dim=1))
        manual = model.head(inp[:, -1]).squeeze(-1)
        ref = model(x)
        assert torch.allclose(manual, ref, atol=1e-4), "Manual unroll does not match nn.LSTM!"
        f_all.append(torch.stack(f_layers, 1).cpu().numpy().astype(np.float16))
        i_all.append(torch.stack(i_layers, 1).cpu().numpy().astype(np.float16))
        preds.append(manual.cpu().numpy())

    return {"forget": np.concatenate(f_all), "input": np.concatenate(i_all),
            "pred_scaled": np.concatenate(preds)}
