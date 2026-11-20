"""Recurrent regressors. Only the recurrent cell differs between experiments.

`RecurrentRegressor(cell="rnn")` is the Simple RNN; `cell="lstm"` is the LSTM
counterpart with the same hidden size, depth, dropout and linear head.

`unrolled_forward` re-implements the stacked recurrence step by step using the
module's own weights. It exists only for the vanishing-gradient diagnostic:
cuDNN/ATen fuse the time loop, so the per-step hidden states that actually feed
the recurrence are not exposed by `nn.RNN`'s output tensor.
"""
from __future__ import annotations

import torch
from torch import nn


class RecurrentRegressor(nn.Module):
    def __init__(self, n_features: int, cell: str = "rnn", hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.cell = cell
        self.hidden = hidden
        self.num_layers = num_layers
        if cell == "rnn":
            self.rnn = nn.RNN(n_features, hidden, num_layers=num_layers,
                              nonlinearity="tanh", batch_first=True, dropout=dropout)
        elif cell == "lstm":
            self.rnn = nn.LSTM(n_features, hidden, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        else:
            raise ValueError("cell must be 'rnn' or 'lstm'")
        self.head = nn.Linear(hidden, 1)  # regression head on the last timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)                    # (batch, time, hidden) of top layer
        return self.head(out[:, -1, :]).squeeze(-1)

    # ------------------------------------------------------------------ #
    # Diagnostic-only manual unroll (eval mode: no inter-layer dropout)
    # ------------------------------------------------------------------ #
    def unrolled_forward(self, x: torch.Tensor):
        """Return (prediction, states) where states[layer][t] is h_t of that layer.

        Every h_t is a node of the autograd graph that feeds h_{t+1}, so after
        `loss.backward()` its `.grad` is the full dLoss/dh_t, including the
        recurrent path. Call `retain_grad()` is done here for convenience.
        """
        batch, steps, _ = x.shape
        layer_input = x
        states: list[list[torch.Tensor]] = []
        for layer in range(self.num_layers):
            w_ih = getattr(self.rnn, f"weight_ih_l{layer}")
            w_hh = getattr(self.rnn, f"weight_hh_l{layer}")
            b = getattr(self.rnn, f"bias_ih_l{layer}") + getattr(self.rnn, f"bias_hh_l{layer}")
            h = x.new_zeros(batch, self.hidden)
            c = x.new_zeros(batch, self.hidden)
            layer_states = []
            for t in range(steps):
                pre = layer_input[:, t, :] @ w_ih.T + h @ w_hh.T + b
                if self.cell == "rnn":
                    h = torch.tanh(pre)
                else:  # PyTorch LSTM gate order: input, forget, cell, output
                    i, f, g, o = pre.chunk(4, dim=1)
                    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                    h = torch.sigmoid(o) * torch.tanh(c)
                h.retain_grad()
                layer_states.append(h)
            states.append(layer_states)
            layer_input = torch.stack(layer_states, dim=1)
        return self.head(layer_input[:, -1, :]).squeeze(-1), states


def build_model(n_features: int, cell: str, hidden: int, num_layers: int,
                dropout: float) -> RecurrentRegressor:
    return RecurrentRegressor(n_features, cell, hidden, num_layers, dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
