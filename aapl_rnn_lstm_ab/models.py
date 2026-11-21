"""Recurrent regression model (vanilla RNN or LSTM): (B, seq_len, F) -> (B,) scaled next-day return."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class SeqModel(nn.Module):
    """Stacked nn.RNN / nn.LSTM followed by a linear head on the last time step.

    cfg.cell is the only thing that changes between the two experiments.
    """

    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        kwargs = dict(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        if cfg.cell == "lstm":
            self.rnn = nn.LSTM(**kwargs)
        else:
            self.rnn = nn.RNN(nonlinearity="tanh", **kwargs)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one prediction per sample, shape (B,)."""
        out, _ = self.rnn(x)               # (B, T, H), same for RNN and LSTM
        last = self.drop(out[:, -1, :])    # (B, H)
        return self.head(last).squeeze(-1)


def build_model(input_size: int, cfg: Config) -> SeqModel:
    """Construct the model from the config."""
    return SeqModel(input_size, cfg)