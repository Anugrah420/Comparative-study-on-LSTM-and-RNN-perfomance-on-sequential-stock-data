"""Data loading, feature engineering, chronological splitting, scaling and DataLoaders.

Design notes
------------
* Target is the NEXT-day log return. A sample at row t uses features from rows
  [t-seq_len+1 .. t] and predicts the return from close[t] to close[t+1].
* Splits are chronological and assigned by the TARGET row t. Val/test windows may
  look back into the previous split for context (past data only, so no leakage).
* Share the cached CSV (outputs/data/) with your comparison partner so both
  runs use byte-identical prices.
* The scaler is fit on training rows only. The target is standardised too, so
  model outputs are in scaled units; use `Scaler.inverse_target` to get raw returns.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from utils import ensure_dir

FEATURE_COLS = ["log_ret", "hl_range", "oc_ret", "log_vol_chg", "ret_mean_5", "ret_std_10"]


# --------------------------------------------------------------------------- #
# Scaler
# --------------------------------------------------------------------------- #
@dataclass
class Scaler:
    """Standardiser for features and target, fit on the training split only."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "Scaler":
        """Compute mean/std from training features x (N, F) and targets y (N,)."""
        return cls(
            x_mean=x.mean(axis=0),
            x_std=np.maximum(x.std(axis=0), 1e-8),
            y_mean=float(y.mean()),
            y_std=float(max(y.std(), 1e-8)),
        )

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        """Standardise features."""
        return (x - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Standardise targets."""
        return (y - self.y_mean) / self.y_std

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to raw next-day log returns."""
        return np.asarray(y_scaled) * self.y_std + self.y_mean


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Sliding-window dataset.

    Attributes
    ----------
    X      : float32 (N, seq_len, F) scaled features
    y      : float32 (N,)            scaled next-day return (training target)
    y_raw  : float64 (N,)            raw next-day log return (for metrics/backtest)
    dates  : DatetimeIndex (N,)      date of the last row in each window (decision day)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, y_raw: np.ndarray, dates: pd.DatetimeIndex):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.y_raw = y_raw
        self.dates = dates

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# --------------------------------------------------------------------------- #
# Loading and features
# --------------------------------------------------------------------------- #
def load_prices(cfg: Config) -> pd.DataFrame:
    """Download OHLCV for cfg.ticker (cached as CSV) and return a date-indexed DataFrame."""
    cache_dir = ensure_dir(f"{cfg.out_dir}/data")
    cache = cache_dir / f"{cfg.ticker}_{cfg.start}_{cfg.end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # imported lazily so the rest works offline with a cache

        df = yf.download(cfg.ticker, start=cfg.start, end=cfg.end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise RuntimeError(f"No data returned for {cfg.ticker}")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        df.to_csv(cache)
    return df.sort_index()


def data_hash(df: pd.DataFrame) -> str:
    """Short hash of the price data, so compare.py can verify both runs used the same data."""
    h = pd.util.hash_pandas_object(df.round(6), index=True).values.tobytes()
    return hashlib.sha256(h).hexdigest()[:12]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of FEATURE_COLS plus 'target' (next-day log return), NaNs dropped."""
    close = df["Close"]
    vol = df["Volume"].replace(0, np.nan).ffill()
    log_ret = np.log(close).diff()

    out = pd.DataFrame(index=df.index)
    out["log_ret"] = log_ret
    out["hl_range"] = np.log(df["High"] / df["Low"])
    out["oc_ret"] = np.log(df["Close"] / df["Open"])
    out["log_vol_chg"] = np.log(vol).diff()
    out["ret_mean_5"] = log_ret.rolling(5).mean()
    out["ret_std_10"] = log_ret.rolling(10).std()
    out["target"] = log_ret.shift(-1)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# --------------------------------------------------------------------------- #
# Splits and loaders
# --------------------------------------------------------------------------- #
def _windows(X_all: np.ndarray, y_scaled: np.ndarray, y_raw: np.ndarray,
             dates: pd.DatetimeIndex, lo: int, hi: int, seq_len: int) -> WindowDataset:
    """Build windows whose target row t lies in [lo, hi)."""
    start = max(lo, seq_len - 1)
    idx = np.arange(start, hi)
    Xw = np.stack([X_all[t - seq_len + 1: t + 1] for t in idx])
    return WindowDataset(Xw, y_scaled[idx], y_raw[idx], dates[idx])


def make_splits(df: pd.DataFrame, cfg: Config):
    """Chronological train/val/test split. Returns (train_ds, val_ds, test_ds, scaler)."""
    feats = build_features(df)
    n = len(feats)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    if n_train <= cfg.seq_len or n - n_train - n_val <= 0:
        raise ValueError("Not enough data for the requested split / seq_len")

    x_raw = feats[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_raw = feats["target"].to_numpy(dtype=np.float64)

    scaler = Scaler.fit(x_raw[:n_train], y_raw[:n_train])   # train rows only
    X_all = scaler.transform_x(x_raw)
    y_scaled = scaler.transform_y(y_raw)
    dates = feats.index

    train = _windows(X_all, y_scaled, y_raw, dates, 0, n_train, cfg.seq_len)
    val = _windows(X_all, y_scaled, y_raw, dates, n_train, n_train + n_val, cfg.seq_len)
    test = _windows(X_all, y_scaled, y_raw, dates, n_train + n_val, n, cfg.seq_len)
    return train, val, test, scaler


def make_loaders(train: Dataset, val: Dataset, test: Dataset, cfg: Config):
    """Wrap datasets in DataLoaders. Only the train loader shuffles (seeded)."""
    g = torch.Generator().manual_seed(cfg.seed)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=g),
        DataLoader(val, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(test, batch_size=cfg.batch_size, shuffle=False),
    )