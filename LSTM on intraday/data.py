"""
data.py — download, clean, feature-engineer, split, scale and window.

This file is meant to be shared VERBATIM with the Simple RNN intraday
experiment, so both models see exactly the same inputs.

Leakage rules enforced here:
  * every indicator is causal (uses bars <= t only);
  * VWAP resets every session;
  * windows never span the overnight gap (window + target bar must sit inside
    one session and be contiguous 5-minute bars);
  * the split is by whole trading days, chronologically (70/15/15);
  * scalers are fit on TRAIN rows only; the target has its own scaler.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as C


# ═════════════════════════════════════════════════════════════════════════
# 1. Download + clean
# ═════════════════════════════════════════════════════════════════════════
def download_intraday(ticker: str = C.TICKER) -> pd.DataFrame:
    """Pull 5-minute OHLCV for AAPL from yfinance (max 60 days)."""
    import yfinance as yf

    raw = yf.download(
        ticker, interval=C.INTERVAL, period=C.PERIOD,
        auto_adjust=False, prepost=False, progress=False, threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data (check internet / ticker).")
    return raw


def clean_intraday(raw: pd.DataFrame, ticker: str = C.TICKER) -> pd.DataFrame:
    """Flatten MultiIndex, drop NaN/duplicates, convert to New York time and
    keep the regular session only (bars starting 09:30 … 15:55)."""
    df = raw.copy()

    # Newer yfinance returns MultiIndex columns like ('Close', 'AAPL').
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(lvl):
                df = df.xs(ticker, axis=1, level=lvl)
                break
        else:
            df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    idx = pd.to_datetime(df.index)
    if idx.tz is None:                       # yfinance intraday is usually tz-aware
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(C.MARKET_TZ)
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.between_time(C.SESSION_START, C.SESSION_LAST_BAR)   # regular session
    return df


def load_data(csv_path: str | None = None, save_cache: bool = True) -> pd.DataFrame:
    """Load from a cached CSV (for exact RNN/LSTM pairing) or download fresh."""
    if csv_path:
        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index, utc=True)
        df = clean_intraday(raw)
        print(f"[data] Loaded cached data from {csv_path}")
    else:
        df = clean_intraday(download_intraday())
        if save_cache:
            C.OUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(C.RAW_CACHE_CSV)
            print(f"[data] Saved raw snapshot -> {C.RAW_CACHE_CSV}")

    n_days = len(np.unique(df.index.date))
    print(f"[data] Rows: {len(df):,} | Range: {df.index.min()} -> {df.index.max()} "
          f"| Trading days: {n_days}")
    return df


# ═════════════════════════════════════════════════════════════════════════
# 2. Feature engineering (causal)
# ═════════════════════════════════════════════════════════════════════════
def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing = EMA with alpha = 1/n (causal)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session"] = out.index.date                    # New York trading date
    grp = out["session"]
    o, h, l, c, v = (out[k] for k in ["Open", "High", "Low", "Close", "Volume"])

    # EMA_9 (causal; state carries across sessions, which is not look-ahead)
    out["EMA_9"] = c.ewm(span=9, adjust=False, min_periods=9).mean()

    # VWAP — resets every session
    tp = (h + l + c) / 3.0
    cum_pv = (tp * v).groupby(grp).cumsum()
    cum_v = v.groupby(grp).cumsum()
    out["VWAP"] = (cum_pv / cum_v.where(cum_v > 0)).fillna(tp)

    # RSI_14 (Wilder)
    delta = c.diff()
    avg_gain = _wilder(delta.clip(lower=0), 14)
    avg_loss = _wilder((-delta).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    out["RSI_14"] = rsi

    # ATR_14 (Wilder)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["ATR_14"] = _wilder(tr, 14)

    # Cyclical minute-of-day (m = minutes since midnight, NY time)
    minutes = np.asarray(out.index.hour * 60 + out.index.minute, dtype=float)
    out["minute_of_day"] = minutes.astype(int)
    out["Minute_of_Day_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    out["Minute_of_Day_cos"] = np.cos(2 * np.pi * minutes / 1440.0)

    # Diagnostics only (NOT model inputs): within-session log-return volatility
    logret = np.log(c).groupby(grp).diff()            # excludes overnight gap
    out["ret_std_12"] = logret.groupby(grp).transform(
        lambda s: s.rolling(C.VOL_STD_WINDOW, min_periods=C.VOL_STD_WINDOW).std()
    )
    return out.dropna(subset=C.FEATURES)


# ═════════════════════════════════════════════════════════════════════════
# 3. Split (by whole trading days, chronological)
# ═════════════════════════════════════════════════════════════════════════
def split_sessions(feat: pd.DataFrame) -> dict:
    days = np.array(sorted(feat["session"].unique()))
    n = len(days)
    n_tr = int(round(n * C.TRAIN_FRAC))
    n_va = int(round(n * C.VAL_FRAC))
    splits = {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}
    for k, d in splits.items():
        print(f"[split] {k:5s}: {len(d):2d} days  ({d[0]} -> {d[-1]})")
    return splits


# ═════════════════════════════════════════════════════════════════════════
# 4. Scaling (fit on TRAIN only)
# ═════════════════════════════════════════════════════════════════════════
@dataclass
class Scalers:
    x: StandardScaler
    y: StandardScaler
    n_train_rows: int
    fit_last_ts: pd.Timestamp


def fit_scalers(feat: pd.DataFrame, train_days) -> Scalers:
    tr = feat[feat["session"].isin(train_days)]
    xs = StandardScaler().fit(tr[C.FEATURES].to_numpy(float))
    ys = StandardScaler().fit(tr[[C.TARGET]].to_numpy(float))
    return Scalers(xs, ys, len(tr), tr.index.max())


# ═════════════════════════════════════════════════════════════════════════
# 5. Windowing (never across the overnight gap)
# ═════════════════════════════════════════════════════════════════════════
@dataclass
class WindowSet:
    X: np.ndarray        # [N, WINDOW, F] scaled inputs
    y: np.ndarray        # [N] scaled next-bar close
    meta: pd.DataFrame   # one row per sample (bar t = last input bar)


def make_windows(feat: pd.DataFrame, scalers: Scalers, days, window: int = C.WINDOW) -> WindowSet:
    sub = feat[feat["session"].isin(days)]
    Xs = scalers.x.transform(sub[C.FEATURES].to_numpy(float)).astype(np.float32)
    ys = scalers.y.transform(sub[[C.TARGET]].to_numpy(float)).ravel().astype(np.float32)
    ts = sub.index
    sess = sub["session"].to_numpy()
    feat_pos = feat.index.get_indexer(ts)            # positions in the full frame
    span_ok = pd.Timedelta(minutes=C.BAR_MINUTES * (window - 1))
    step_ok = pd.Timedelta(minutes=C.BAR_MINUTES)

    X_out, y_out, ends, tgts = [], [], [], []
    for d in days:
        idx = np.flatnonzero(sess == d)
        if len(idx) < window + 1:
            continue                                  # e.g. early-close half days
        for e in range(window - 1, len(idx) - 1):
            s = e - window + 1
            if ts[idx[e]] - ts[idx[s]] != span_ok or ts[idx[e + 1]] - ts[idx[e]] != step_ok:
                continue                              # missing bar inside window
            X_out.append(Xs[idx[s:e + 1]])
            y_out.append(ys[idx[e + 1]])
            ends.append(idx[e])
            tgts.append(idx[e + 1])

    ends, tgts = np.array(ends, int), np.array(tgts, int)
    meta = pd.DataFrame({
        "timestamp": ts[ends],                        # bar t (decision bar)
        "target_timestamp": ts[tgts],                 # bar t+1
        "session": sess[ends],
        "close_t": sub["Close"].to_numpy()[ends],
        "actual_next": sub["Close"].to_numpy()[tgts],
        "vwap_t": sub["VWAP"].to_numpy()[ends],
        "atr_t": sub["ATR_14"].to_numpy()[ends],
        "ret_std_12_t": sub["ret_std_12"].to_numpy()[ends],
        "row_end": feat_pos[ends],                    # position of bar t in `feat`
    })
    X = np.stack(X_out) if X_out else np.empty((0, window, len(C.FEATURES)), np.float32)
    return WindowSet(X, np.array(y_out, np.float32), meta)


# ═════════════════════════════════════════════════════════════════════════
# 6. Leakage assertions
# ═════════════════════════════════════════════════════════════════════════
def assert_no_leakage(clean: pd.DataFrame, feat: pd.DataFrame, splits: dict, scalers: Scalers):
    """(1) Indicators are causal: recomputing them on data truncated at the end
    of TRAIN gives identical values on TRAIN rows. (2) Scalers were fit on
    TRAIN rows only, all of which precede every val/test bar."""
    train_rows = feat[feat["session"].isin(splits["train"])]
    first_val = feat[feat["session"].isin(splits["val"])].index.min()
    first_test = feat[feat["session"].isin(splits["test"])].index.min()

    trunc = build_features(clean[clean.index <= train_rows.index.max()])
    a = feat.loc[trunc.index, C.FEATURES].to_numpy(float)
    b = trunc[C.FEATURES].to_numpy(float)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9), "Indicator look-ahead detected!"

    assert scalers.x.n_samples_seen_ == len(train_rows), "X scaler saw non-train rows!"
    assert scalers.y.n_samples_seen_ == len(train_rows), "y scaler saw non-train rows!"
    assert scalers.fit_last_ts < first_val < first_test, "Scaler fit window overlaps val/test!"
    print("[leakage] OK — indicators causal; scalers fit on TRAIN only "
          f"({scalers.n_train_rows:,} rows, last fit bar {scalers.fit_last_ts}).")


# ═════════════════════════════════════════════════════════════════════════
# 7. One-call pipeline
# ═════════════════════════════════════════════════════════════════════════
def prepare(csv_path: str | None = None) -> dict:
    clean = load_data(csv_path)
    feat = build_features(clean)
    splits = split_sessions(feat)
    scalers = fit_scalers(feat, splits["train"])
    assert_no_leakage(clean, feat, splits, scalers)
    ws = {k: make_windows(feat, scalers, d) for k, d in splits.items()}
    for k, w in ws.items():
        print(f"[windows] {k:5s}: X={w.X.shape}  y={w.y.shape}")
    test_bars = feat[feat["session"].isin(splits["test"])].copy()
    return dict(clean=clean, feat=feat, splits=splits, scalers=scalers,
                windows=ws, test_bars=test_bars)
