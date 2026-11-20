"""Data acquisition and leakage-free feature engineering for AAPL daily data.

Every indicator here is *causal*: the value on date t is computed only from
rows dated <= t (trailing rolling windows and recursive EWMs with
``adjust=False``). ``walkforward.assert_no_lookahead`` re-checks this on every
fold by recomputing the features on truncated raw data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICKER = "AAPL"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Model inputs, in a fixed order shared by the RNN and LSTM experiments.
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_50", "SMA_200", "MACD", "MACD_Signal",
    "Rolling_Std_20", "Month_sin", "Month_cos",
]
# Column groups used by the window-relative normalisation in walkforward.py.
PRICE_LEVEL_COLS = ["Open", "High", "Low", "Close", "SMA_50", "SMA_200"]  # USD levels
PRICE_DIFF_COLS = ["MACD", "MACD_Signal"]                                 # USD differences
VOLUME_COLS = ["Volume"]


# --------------------------------------------------------------------------- #
# Download / loading
# --------------------------------------------------------------------------- #
def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance/CSV frame to a sorted, de-duplicated OHLCV frame."""
    if raw is None or len(raw) == 0:
        raise ValueError("empty price frame")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) columns; keep the field level.
        level = 0 if "Close" in df.columns.get_level_values(0) else 1
        df.columns = df.columns.get_level_values(level)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.loc[:, OHLCV]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df.index.name = "Date"
    return df.astype(float)


def _span_years(df: pd.DataFrame) -> float:
    return (df.index[-1] - df.index[0]).days / 365.25


def download_aapl(years: int = 20, today: pd.Timestamp | None = None):
    """Download ~`years` of daily AAPL bars with the fallbacks the spec requires.

    1. ``period="20y"`` (not an officially documented yfinance period string).
    2. Explicit ``start = today - 20y``, ``end = today``.
    3. ``period="max"`` truncated to the last 20 years.
    Returns (frame, description of the attempt that succeeded, attempt log).
    """
    import yfinance as yf

    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    start = today - pd.DateOffset(years=years)
    tolerance = 0.05  # years; the first trading day can fall a few days after `start`
    log: list[str] = []

    def accept(df: pd.DataFrame, label: str):
        span = _span_years(df)
        log.append(f"{label}: {len(df)} rows, {span:.2f} years")
        return span >= years - tolerance

    try:
        df = _clean(yf.download(TICKER, period=f"{years}y", interval="1d",
                                auto_adjust=True, progress=False))
        if accept(df, f'period="{years}y"'):
            return df, f'yfinance period="{years}y"', log
    except Exception as exc:  # unsupported period string, network error, ...
        log.append(f'period="{years}y" failed: {exc!r}')

    try:
        # yfinance treats `end` as exclusive, so add one day to include today.
        df = _clean(yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                                end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                interval="1d", auto_adjust=True, progress=False))
        if accept(df, "explicit start/end"):
            return df, "yfinance explicit start/end", log
    except Exception as exc:
        log.append(f"explicit start/end failed: {exc!r}")

    try:
        df = _clean(yf.download(TICKER, period="max", interval="1d",
                                auto_adjust=True, progress=False))
        df = df[df.index >= start]
        if accept(df, 'period="max" truncated'):
            return df, 'yfinance period="max" truncated to 20y', log
    except Exception as exc:
        log.append(f'period="max" failed: {exc!r}')

    raise RuntimeError("Could not download 20 years of AAPL data:\n  " + "\n  ".join(log))


def load_csv(path: str) -> pd.DataFrame:
    """Load AAPL OHLCV from a CSV (Date index + Open/High/Low/Close/Volume)."""
    return _clean(pd.read_csv(path, index_col=0, parse_dates=True))


def synthetic_aapl_like(years: int = 20, today: pd.Timestamp | None = None,
                        seed: int = 42) -> pd.DataFrame:
    """OFFLINE SMOKE-TEST DATA ONLY - never use for conclusions about AAPL.

    Regime-switching geometric Brownian motion with crash regimes placed at the
    2008 GFC, the 2020 COVID crash and the 2022 bear market so that every code
    path (crash windows, annotations) is exercised without network access.
    """
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    idx = pd.bdate_range(today - pd.DateOffset(years=years), today)
    drift = np.full(len(idx), 0.0012)
    vol = np.full(len(idx), 0.017)
    for lo, hi, mu, sd in [("2008-09-01", "2009-03-09", -0.004, 0.045),
                           ("2020-02-20", "2020-03-23", -0.012, 0.060),
                           ("2022-01-03", "2022-12-30", -0.0012, 0.025)]:
        m = (idx >= lo) & (idx <= hi)
        drift[m], vol[m] = mu, sd
    close = 3.0 * np.exp(np.cumsum(rng.normal(drift, vol)))
    open_ = close * (1 + rng.normal(0, 0.006, len(idx)))
    wick = np.abs(rng.normal(0, 0.008, len(idx))) * close
    df = pd.DataFrame({
        "Open": open_, "High": np.maximum(open_, close) + wick,
        "Low": np.minimum(open_, close) - wick, "Close": close,
        "Volume": rng.lognormal(18.5, 0.4, len(idx)),
    }, index=idx)
    df.index.name = "Date"
    return df


def coverage_report(df: pd.DataFrame) -> dict:
    years = set(df.index.year)
    return {
        "rows": int(len(df)),
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "span_years": round(_span_years(df), 2),
        "covers_2008": 2008 in years, "covers_2020": 2020 in years,
    }


# --------------------------------------------------------------------------- #
# Feature engineering (pandas only, causal)
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add indicators and the next-day-close target; drop warm-up NaNs.

    Returns the feature frame (FEATURE_COLS + 'Target') and a summary dict.
    """
    out = df[OHLCV].copy()
    close = out["Close"]

    out["SMA_50"] = close.rolling(50, min_periods=50).mean()
    out["SMA_200"] = close.rolling(200, min_periods=200).mean()
    # adjust=False gives the recursive EMA: value at t depends only on rows <= t.
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    daily_ret = close / close.shift(1) - 1.0
    out["Rolling_Std_20"] = daily_ret.rolling(20, min_periods=20).std()
    month = out.index.month
    out["Month_sin"] = np.sin(2 * np.pi * month / 12)
    out["Month_cos"] = np.cos(2 * np.pi * month / 12)

    # Target: next trading day's close. The final row has no next close.
    out["Target"] = close.shift(-1)
    raw_rows = len(out)
    out = out.iloc[:-1]
    before = len(out)
    out = out.dropna()
    info = {
        "raw_rows": int(raw_rows),
        "final_row_dropped": 1,
        "warmup_rows_dropped": int(before - len(out)),
        "usable_rows": int(len(out)),
        "first_usable_date": str(out.index[0].date()),
        "last_usable_date": str(out.index[-1].date()),
    }
    return out[FEATURE_COLS + ["Target"]], info
