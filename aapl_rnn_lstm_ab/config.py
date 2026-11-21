"""Experiment configuration for the RNN/LSTM stock-return experiment.

Config is immutable. Use dataclasses.replace(cfg, lr=5e-4) to vary a value.
`fingerprint()` hashes the settings that must match between runs being
compared. `cell` is deliberately NOT in that list: it is the one thing that
is supposed to differ between the RNN and the LSTM run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Settings both runs must share for a fair comparison (everything except `cell`).
_SHARED = ("ticker", "start", "end", "train_frac", "val_frac", "seq_len",
           "batch_size", "epochs", "lr", "weight_decay", "patience",
           "hidden_size", "num_layers", "dropout", "cost_bps", "seed")


@dataclass(frozen=True)
class Config:
    # model type: the ONLY difference between the two compared runs
    cell: str = "rnn"               # "rnn" | "lstm"

    # data
    ticker: str = "AAPL"
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    train_frac: float = 0.70
    val_frac: float = 0.15          # test = remainder (chronological split)
    seq_len: int = 30

    # training
    batch_size: int = 64
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 10              # early stopping on val loss

    # model
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.3

    # backtest
    cost_bps: float = 5.0           # transaction cost per position change, in basis points

    # misc
    seed: int = 42
    device: str = "auto"            # "auto" | "cpu" | "cuda" | "mps"
    out_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.cell not in ("rnn", "lstm"):
            raise ValueError("cell must be 'rnn' or 'lstm'")
        if not 0 < self.train_frac + self.val_frac < 1:
            raise ValueError("train_frac + val_frac must be in (0, 1)")

    @property
    def run_dir(self) -> Path:
        """Where this run's results are saved: <out_dir>/<cell>/."""
        return Path(self.out_dir) / self.cell

    def shared_settings(self) -> dict:
        """Settings that must match across runs being compared."""
        return {k: getattr(self, k) for k in _SHARED}

    def fingerprint(self) -> str:
        """Short hash of the shared settings (same for RNN and LSTM runs)."""
        blob = json.dumps(self.shared_settings(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]