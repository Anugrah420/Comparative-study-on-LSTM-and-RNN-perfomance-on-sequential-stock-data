"""Metrics (v5). Divide-by-zero -> NaN (JSON null). Profit factor = daily gross profit / daily gross loss (continuous sizing has no discrete trades)."""
import json
import numpy as np
import pandas as pd
import config as C


def safe_div(a, b):
    return float(a / b) if (b is not None and np.isfinite(b) and b != 0) else float("nan")


def fmt_inr(x, decimals=0):
    """Indian digit grouping: 1234567.8 -> Rs 12,34,568."""
    if x is None or not np.isfinite(x):
        return "n/a"
    s = f"{abs(x):.{decimals}f}"; ip, _, fp = s.partition(".")
    if len(ip) > 3:
        head, tail, grp = ip[:-3], ip[-3:], []
        while len(head) > 2:
            grp.insert(0, head[-2:]); head = head[:-2]
        if head: grp.insert(0, head)
        ip = ",".join(grp + [tail])
    return ("-" if x < 0 else "") + "₹" + ip + ("." + fp if fp else "")


def ml_metrics(actual, pred, close_t):
    err = pred - actual
    da_mask = np.sign(actual - close_t) != 0
    da = np.mean(np.sign(pred - close_t)[da_mask] == np.sign(actual - close_t)[da_mask]) * 100
    return {"Directional Accuracy (%)": float(da), "RMSE (USD)": float(np.sqrt(np.mean(err ** 2))),
            "MAPE (%)": float(np.mean(np.abs(err) / np.abs(actual)) * 100)}


def max_drawdown_pct(eq):
    e = np.r_[C.INITIAL_CAPITAL, np.asarray(eq)]
    return float((e / np.maximum.accumulate(e) - 1).min() * 100)


def trading_metrics(ret, eq, pnls, n_trades=None, exposure=None):
    ret, eq, pnls = np.asarray(ret), np.asarray(eq), np.asarray(pnls)
    net = float(eq[-1] - C.INITIAL_CAPITAL)
    wins, losses = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    ex = ret - C.RF_ANNUAL / C.TRADING_DAYS
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    p5 = float(np.percentile(ret, 5)); mdd = max_drawdown_pct(eq)
    cagr = ((eq[-1] / C.INITIAL_CAPITAL) ** (C.TRADING_DAYS / len(ret)) - 1) * 100
    return {"Total ROI (%)": net / C.INITIAL_CAPITAL * 100, "CAGR (%)": float(cagr), "Net Profit (INR)": net,
            "Profit Factor": safe_div(wins, losses), "Total Trades": int(n_trades if n_trades is not None else len(pnls)),
            "Avg Exposure (%)": float("nan") if exposure is None else exposure * 100,
            "MDD (%)": mdd, "Calmar": safe_div(cagr, abs(mdd)), "VaR95 (%)": p5 * 100, "VaR95 (INR)": p5 * C.INITIAL_CAPITAL,
            "Sharpe": safe_div(ex.mean(), ex.std(ddof=1)) * np.sqrt(C.TRADING_DAYS),
            "Sortino": safe_div(ex.mean(), dd) * np.sqrt(C.TRADING_DAYS)}


def full_matrix(oos, bt, pnl, ntr, bh_pnl):
    a, p, c = oos["Actual_Next"].values, oos["Pred_Next"].values, oos["Close"].values
    strat = {**trading_metrics(bt["strat_ret"], bt["equity"], pnl, ntr, float(bt["position"].mean())), **ml_metrics(a, p, c)}
    bh = trading_metrics(bt["bh_ret"], bt["bh_equity"], bh_pnl, 1, 1.0)
    pers = ml_metrics(a, c, c); pers["Directional Accuracy (%)"] = float("nan")   # predicts "no change"
    return {"strategy": strat, "buy_hold": bh, "persistence": pers}


def rule_metrics(rules):
    return {k: trading_metrics(v["ret"], v["equity"], v["pnl"], v["n_trades"], v["exposure"]) for k, v in rules.items()}


def exposure_matched(bt, exposure):
    """Constant fraction `exposure` of equity in the asset every day (rest cash, 0 %), no rebalancing costs.
    Same Sharpe as Buy & Hold (scale-invariant), so it isolates what 'being invested less' alone achieves."""
    r = exposure * bt["asset_ret"].values
    eq = C.INITIAL_CAPITAL * np.cumprod(1.0 + r)
    m = trading_metrics(r, eq, np.diff(np.r_[C.INITIAL_CAPITAL, eq]), 0, exposure)
    m["Total Trades"] = float("nan")
    return m


def sharpe_diff_bootstrap(rs, rb, n_boot=2000, block=21, seed=None):
    """Paired circular block bootstrap of (Sharpe_strategy - Sharpe_benchmark); block keeps volatility clustering."""
    rs, rb = np.asarray(rs), np.asarray(rb); T = len(rs); rng = np.random.default_rng(C.SEED if seed is None else seed)
    sh = lambda x: (x.mean() / x.std(ddof=1) * np.sqrt(C.TRADING_DAYS)) if x.std(ddof=1) > 0 else np.nan
    nb = int(np.ceil(T / block)); d = np.empty(n_boot)
    for i in range(n_boot):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        d[i] = sh(rs[idx]) - sh(rb[idx])
    return {"point": float(sh(rs) - sh(rb)), "ci_lo": float(np.nanpercentile(d, 2.5)), "ci_hi": float(np.nanpercentile(d, 97.5)),
            "p_diff_le_0": float(np.nanmean(d <= 0))}


def skill_diagnostics(oos, bt):
    """Does the model predict anything? IC = corr(predicted return, actual return); also corr(exposure, next-day return)."""
    act = np.log(oos["Actual_Next"].values / oos["Close"].values)
    pred = oos["Pred_Ret"].values if "Pred_Ret" in oos else np.log(oos["Pred_Next"].values / oos["Close"].values)
    s = pd.DataFrame({"p": pred, "a": act, "w": bt["position"].values})
    hi = bt["position"].values > 0.5; ar = bt["asset_ret"].values
    return {"IC_pearson": float(s["p"].corr(s["a"])), "IC_spearman": float(s["p"].rank().corr(s["a"].rank())),   # Spearman = Pearson on ranks (no scipy)
            "corr_exposure_vs_next_ret": float(s["w"].corr(s["a"])),
            "base_rate_up_days_%": float((act > 0).mean() * 100), "pred_up_share_%": float((pred > 0).mean() * 100),
            "avg_daily_ret_when_exposure>0.5_bps": float(ar[hi].mean() * 1e4) if hi.any() else float("nan"),
            "avg_daily_ret_when_exposure<=0.5_bps": float(ar[~hi].mean() * 1e4) if (~hi).any() else float("nan")}


ORDER = ["Total ROI (%)", "CAGR (%)", "Net Profit (INR)", "Directional Accuracy (%)", "Profit Factor", "Total Trades", "Avg Exposure (%)",
         "MDD (%)", "Calmar", "VaR95 (%)", "VaR95 (INR)", "Sharpe", "Sortino", "RMSE (USD)", "MAPE (%)"]


def _fmt(k, v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "n/a"
    if "INR" in k: return fmt_inr(v)
    if k == "Total Trades": return str(int(v))
    return f"{v:,.3f}" if k in ("Sharpe", "Sortino", "Calmar") else f"{v:,.2f}"


def matrix_table(cols):
    """cols: {column title: metric dict} -> display DataFrame (INR in Indian grouping)."""
    df = pd.DataFrame({t: [_fmt(k, m.get(k)) for k in ORDER] for t, m in cols.items()}, index=ORDER)
    df.index.name = "Metric (VaR/MDD negative = loss)"; return df


def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if hasattr(o, "isoformat"): return str(o)
    return o


def save_json(obj, path):
    with open(path, "w") as f: json.dump(jsonable(obj), f, indent=2, default=str)
