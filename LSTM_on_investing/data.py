"""Data download + feature engineering. Same logic as the RNN long-term experiment's data.py."""
import os
import numpy as np
import pandas as pd
import config as C

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):           # yfinance >=0.2.4x returns (field, ticker)
        df = df.copy(); df.columns = df.columns.get_level_values(0)
    df = df[OHLCV]
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def _ok(df):
    return df is not None and len(df) > 0 and (df.index.max() - df.index.min()).days >= 365.25 * 19.5


def download_prices(cache_path):
    """Try period='20y'; fall back to explicit start/end, then period='max' truncated; then local cache."""
    import yfinance as yf
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=20)
    attempts = [
        ('period="20y"', lambda: yf.download(C.TICKER, period="20y", interval="1d", auto_adjust=True, progress=False)),
        ("explicit start/end", lambda: yf.download(C.TICKER, start=start.strftime("%Y-%m-%d"),
                                                   end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                                   interval="1d", auto_adjust=True, progress=False)),
        ('period="max" truncated', lambda: yf.download(C.TICKER, period="max", interval="1d",
                                                        auto_adjust=True, progress=False)),
    ]
    df = None
    for name, fn in attempts:
        try:
            d = _clean(fn())
            if name.endswith("truncated"):
                d = d[d.index >= start]
            if _ok(d):
                df = d; print(f"[data] downloaded via {name}: {d.index.min().date()} -> {d.index.max().date()}"); break
            print(f"[data] {name} returned <20y of data, trying next fallback")
        except Exception as e:                                       # noqa: BLE001
            print(f"[data] {name} failed: {e}")
    if df is None and os.path.exists(cache_path):
        df = _clean(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        print(f"[data] using cached file {cache_path}")
    if df is None or len(df) == 0:
        raise RuntimeError(f"Could not obtain {C.TICKER} data from yfinance or cache.")
    df.to_csv(cache_path)
    return df


def verify_coverage(df):
    for yr in (2008, 2020):
        n = int((df.index.year == yr).sum())
        assert n > 200, f"Year {yr} not covered adequately ({n} rows)"
    print(f"[data] OK: {len(df)} rows, 2008 and 2020 covered.")


EXOG = {"VIX": "^VIX", "TNX": "^TNX", "HYG": "HYG", "LQD": "LQD", "TLT": "TLT", "GLD": "GLD"}


def load_exog(index, cache_dir):
    """Download macro / cross-asset series and align to the trading calendar by forward-fill (past values only).
    Any series that fails (or starts too late) is skipped, and the features that need it are dropped."""
    import yfinance as yf
    start, end = index.min() - pd.Timedelta(days=10), index.max() + pd.Timedelta(days=3)
    out = {}
    for name, t in EXOG.items():
        path = os.path.join(cache_dir, f"exog_{C.safe_name(t)}.csv"); s = None
        try:
            df = yf.download(t, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            s = df["Close"].dropna()
        except Exception as e:                                           # noqa: BLE001
            print(f"[exog] {t} download failed: {e}")
        if (s is None or len(s) == 0) and os.path.exists(path):
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]; print(f"[exog] {t}: using cache")
        if s is None or len(s) == 0:
            print(f"[exog] {t}: unavailable -> its features are dropped"); continue
        s = s[~s.index.duplicated(keep="first")].sort_index(); s.index = pd.to_datetime(s.index).tz_localize(None)
        s.to_csv(path)
        s = s.reindex(s.index.union(index)).ffill().reindex(index)      # forward-fill only
        if s.first_valid_index() is None or s.first_valid_index() > index[0] + pd.Timedelta(days=400):
            print(f"[exog] {t}: history starts too late -> dropped"); continue
        out[name] = s
    print(f"[exog] using: {list(out)}")
    return pd.DataFrame(out, index=index)


def build_features(raw, exog=None):
    """v5: stationary, causal features. Target = next-day log return (label only). Sets C.FEATURES / C.FEATURE_GROUPS."""
    d = raw[OHLCV].copy()
    c = d["Close"]; logc = np.log(c); ret = logc.diff()
    d["SMA_50"] = c.rolling(50).mean(); d["SMA_200"] = c.rolling(200).mean()
    # --- momentum / trend
    for n in (5, 21, 63, 126): d[f"Ret_{n}d"] = logc.diff(n)
    d["Ret_1d"] = ret
    d["Mom_12_1"] = logc.shift(21) - logc.shift(252)                      # 12-month momentum skipping the last month
    d["DD_252"] = c / c.rolling(252).max() - 1                            # distance from 1-year high
    d["Close_SMA50"] = c / d["SMA_50"] - 1; d["Close_SMA200"] = c / d["SMA_200"] - 1
    d["SMA50_SMA200"] = d["SMA_50"] / d["SMA_200"] - 1
    d["SMA200_Slope"] = d["SMA_200"] / d["SMA_200"].shift(20) - 1
    # --- volatility / regime (from price)
    d["Std_20"] = ret.rolling(20).std(); d["Std_60"] = ret.rolling(60).std(); d["Vol_Ratio"] = d["Std_20"] / d["Std_60"]
    d["Down_Semi_20"] = np.sqrt((ret.clip(upper=0) ** 2).rolling(20).mean())
    d["Park_20"] = np.sqrt((np.log(d["High"] / d["Low"]) ** 2).rolling(20).mean() / (4 * np.log(2)))
    d["HL_Range"] = np.log(d["High"] / d["Low"])
    # --- candle / volume / oscillators
    d["Gap"] = np.log(d["Open"] / c.shift(1)); d["Intraday"] = np.log(c / d["Open"])
    vol = d["Volume"].clip(lower=1); d["Vol_Rel"] = np.log(vol / vol.rolling(20).mean())
    dl = c.diff(); g = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); l = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI_14"] = (100 - 100 / (1 + g / l.replace(0, 1e-12)) - 50) / 50
    d["BB_Z"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; d["MACD_N"] = macd / c; d["MACD_Sig_N"] = macd.ewm(span=9, adjust=False).mean() / c
    m = d.index.month; d["Month_sin"] = np.sin(2 * np.pi * m / 12); d["Month_cos"] = np.cos(2 * np.pi * m / 12)
    base = ["Ret_1d", "Ret_5d", "Ret_21d", "Ret_63d", "Ret_126d", "Mom_12_1", "DD_252", "Close_SMA50", "Close_SMA200", "SMA50_SMA200",
            "SMA200_Slope", "Std_20", "Std_60", "Vol_Ratio", "Down_Semi_20", "Park_20", "HL_Range", "Gap", "Intraday", "Vol_Rel",
            "RSI_14", "BB_Z", "MACD_N", "MACD_Sig_N", "Month_sin", "Month_cos"]
    groups = {"trend200": ["Close_SMA200", "SMA200_Slope", "SMA50_SMA200"],
              "momentum": ["Ret_5d", "Ret_21d", "Ret_63d", "Ret_126d", "Mom_12_1", "DD_252"],
              "vol_regime": ["Std_20", "Std_60", "Vol_Ratio", "Down_Semi_20", "Park_20", "HL_Range"]}
    # --- exogenous: VIX, rates, credit, safe havens (each block only if its series exists)
    ex = exog if exog is not None else pd.DataFrame(index=d.index)
    ex = ex.reindex(d.index)
    vix_f, macro_f = [], []
    if "VIX" in ex:
        v = ex["VIX"]; d["VIX_Log"] = np.log(v); d["VIX_Chg5"] = v / v.shift(5) - 1; d["VIX_Rel"] = v / v.rolling(60).mean() - 1
        d["VRP"] = np.log(v / 100) - np.log(d["Std_20"] * np.sqrt(252) + 1e-9)       # implied minus realised vol
        vix_f = ["VIX_Log", "VIX_Chg5", "VIX_Rel", "VRP"]
    if "TNX" in ex:
        t = ex["TNX"]; d["TNX_Chg21"] = t / t.shift(21) - 1; d["TNX_Chg63"] = t / t.shift(63) - 1; macro_f += ["TNX_Chg21", "TNX_Chg63"]
    if "HYG" in ex and "LQD" in ex:
        q = ex["HYG"] / ex["LQD"]; d["Credit_Mom21"] = np.log(q / q.shift(21)); d["Credit_Rel"] = q / q.rolling(126).mean() - 1
        macro_f += ["Credit_Mom21", "Credit_Rel"]
    if "TLT" in ex: d["TLT_Ret21"] = np.log(ex["TLT"] / ex["TLT"].shift(21)); macro_f.append("TLT_Ret21")
    if "GLD" in ex: d["GLD_Ret21"] = np.log(ex["GLD"] / ex["GLD"].shift(21)); macro_f.append("GLD_Ret21")
    if vix_f: groups["vix"] = vix_f
    if macro_f: groups["macro_credit"] = macro_f
    C.FEATURES = base + vix_f + macro_f; C.FEATURE_GROUPS = groups
    d["Target_Ret"] = logc.shift(-1) - logc                              # label only, never an input
    d["Target_Close"] = c.shift(-1)
    d["Next_Date"] = pd.Series(d.index, index=d.index).shift(-1)
    return d.dropna(subset=C.FEATURES + ["Target_Ret", "Target_Close", "Next_Date", "SMA_200"])


def assert_no_lookahead(raw, exog=None):
    """Causality test: features for rows <= k computed on truncated data == computed on full data."""
    full = build_features(raw, exog)
    for k in (len(raw) // 2, len(raw) - 300):
        part = build_features(raw.iloc[:k], None if exog is None else exog.iloc[:k])
        idx = part.index[:-1].intersection(full.index)
        assert len(idx) > 100
        assert np.allclose(part.loc[idx, C.FEATURES].values, full.loc[idx, C.FEATURES].values,
                           equal_nan=True), "Look-ahead leakage detected in feature engineering!"
    print("[data] causality assertion passed (features unchanged when future data is removed).")


def make_synthetic(seed=0):
    """OFFLINE SMOKE-TEST DATA ONLY (not a real asset). Regime-switching GBM with 2008/2020-style crashes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=20 * 252 + 20)
    r = rng.normal(0.0006, 0.017, len(idx))
    for yr, (a, b) in {2008: (0.1, 0.5), 2020: (0.1, 0.2), 2022: (0.0, 0.8)}.items():
        pos = np.where(idx.year == yr)[0]
        if len(pos): r[pos[int(a * len(pos)):int(b * len(pos))]] -= 0.004
    close = 5 * np.exp(np.cumsum(r))
    o = close * (1 + rng.normal(0, .004, len(idx)))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, close) * 1.005, "Low": np.minimum(o, close) * .995,
                         "Close": close, "Volume": rng.integers(1e6, 5e7, len(idx)).astype(float)}, index=idx)


download_aapl = download_prices   # backwards-compatible alias
