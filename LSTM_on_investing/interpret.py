"""Crash zones, gate/cell-state vs 200-day-trend tests, crash-window comparison, SMA_200 ablation, verdict."""
import numpy as np
import pandas as pd
import config as C
from backtest import run_backtest
from metrics import ml_metrics, max_drawdown_pct, trading_metrics


# ---------- crash zones (programmatic) ----------
def find_episodes(close):
    peak = close.cummax(); dd = close / peak - 1
    grp = (dd == 0).cumsum(); eps = []
    for g, sub in dd.groupby(grp):
        depth = sub.min()
        if depth <= -C.DD_MIN_DEPTH:
            tr = sub.idxmin(); pk = sub.index[0]
            eps.append(dict(peak=pk, trough=tr, depth=float(depth), label=_label(pk, tr)))
    eps = sorted(eps, key=lambda e: e["depth"])[:C.DD_TOP_K]
    return sorted(eps, key=lambda e: e["peak"])


def _label(pk, tr):
    if pk.year in (2007, 2008) and tr.year in (2008, 2009): return "2008 GFC"
    if pk.year == 2020 and tr.year == 2020: return "2020 COVID"
    if pk.year == 2022 or tr.year == 2022: return "2022 bear"
    return f"DD {pk:%b%y}-{tr:%b%y}"


# ---------- (a) gates / cell state vs 200-day trend ----------
def _corr(a, b, method="pearson"):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if method == "spearman":
        s = s.rank()                                                    # Spearman = Pearson on ranks (no scipy needed)
    return float(s["a"].corr(s["b"])) if len(s) > 10 and s["a"].std() > 0 and s["b"].std() > 0 else float("nan")


def trend_vars(feat, idx):
    sma = feat["SMA_200"]
    return {"Close/SMA200-1": (feat["Close"] / sma - 1).reindex(idx).values,
            "SMA200 slope(20d)": (sma / sma.shift(20) - 1).reindex(idx).values,
            "[control] Close/SMA50-1": feat["Close_SMA50"].reindex(idx).values,       # placebo: shorter trend
            "[control] 21d return": feat["Ret_21d"].reindex(idx).values}


def gate_analysis(gates, oos, feat):
    idx, fold = oos.index, oos["Fold"].values
    tv = trend_vars(feat, idx)
    fm_top, fm_l1 = gates["last_f"][:, -1].mean(1), gates["last_f"][:, 0].mean(1)
    res = {"pooled": {}, "within_fold": {}, "first_diff": {}, "pc1_per_fold": [], "best_unit_per_fold": []}
    for name, v in tv.items():
        for lname, fm in (("forget_top", fm_top), ("forget_l1", fm_l1)):
            key = f"{lname} vs {name}"
            res["pooled"][key] = {"pearson": _corr(fm, v), "spearman": _corr(fm, v, "spearman")}
            df = pd.DataFrame({"f": fm, "v": v, "k": fold})
            dm = df[["f", "v"]] - df.groupby("k")[["f", "v"]].transform("mean")
            res["within_fold"][key] = _corr(dm["f"], dm["v"])
            d = df.groupby("k")[["f", "v"]].diff()
            res["first_diff"][key] = _corr(d["f"], d["v"])
    for f in np.unique(fold):
        m = fold == f; Cm = gates["last_c"][m, -1]; Cc = Cm - Cm.mean(0)
        pc1 = Cc @ np.linalg.svd(Cc, full_matrices=False)[2][0]
        row_pc, row_bu = {"fold": int(f)}, {"fold": int(f)}
        for name, v in tv.items():
            row_pc[name] = abs(_corr(pc1, v[m]))                         # sign of a PC is arbitrary -> |r|
            ucs = [abs(_corr(Cm[:, j], v[m])) for j in range(Cm.shape[1])]
            ucs = np.nan_to_num(ucs); row_bu[name] = float(ucs.max()); row_bu[name + " unit"] = int(ucs.argmax())
        res["pc1_per_fold"].append(row_pc); res["best_unit_per_fold"].append(row_bu)
    for name in tv:
        res.setdefault("median_abs_pc1", {})[name] = float(np.nanmedian([r[name] for r in res["pc1_per_fold"]]))
        res.setdefault("median_abs_best_unit", {})[name] = float(np.nanmedian([r[name] for r in res["best_unit_per_fold"]]))
    res["forget_top_series"], res["forget_l1_series"] = fm_top, fm_l1
    return res


# ---------- (b) crash windows ----------
def _sub_stats(r):
    r = np.asarray(r); eq = C.INITIAL_CAPITAL * np.cumprod(1 + r)
    return {"return_%": float((eq[-1] / C.INITIAL_CAPITAL - 1) * 100), "MDD_%": max_drawdown_pct(eq)}


def crash_table(bt, episodes, oos_start, rnn_eq=None, exposure=None):
    """EM = exposure-matched buy&hold (constant weight = strategy's average exposure). 'flat' day = exposure < 0.5."""
    rows = []
    for e in episodes:
        lo, hi = max(e["peak"], oos_start), e["trough"]
        w = bt.loc[(bt.index >= lo) & (bt.index <= hi)]
        row = {"window": e["label"], "peak": e["peak"].date(), "trough": e["trough"].date(),
               "B&H full-DD_%": e["depth"] * 100, "oos_days": len(w)}
        if len(w) < C.MIN_CRASH_DAYS:
            row["note"] = "in-sample / too few OOS days - not scored"; rows.append(row); continue
        s, b = _sub_stats(w["strat_ret"]), _sub_stats(w["bh_ret"])
        row.update({"LSTM_ret_%": s["return_%"], "LSTM_MDD_%": s["MDD_%"], "LSTM_avg_w_%": float(w["position"].mean() * 100),
                    "LSTM_days_flat": int((w["position"] < 0.5).sum()), "LSTM_flat_%": float((w["position"] < 0.5).mean() * 100),
                    "BH_ret_%": b["return_%"], "BH_MDD_%": b["MDD_%"]})
        if exposure is not None:
            em = _sub_stats(exposure * w["asset_ret"]); row.update({"EM_ret_%": em["return_%"], "EM_MDD_%": em["MDD_%"]})
        if rnn_eq is not None:
            ww = rnn_eq.loc[(rnn_eq.index >= lo) & (rnn_eq.index <= hi)]
            if len(ww) > 2:
                rs = _sub_stats(ww["equity"].pct_change().fillna(0.0)); row.update({"RNN_ret_%": rs["return_%"], "RNN_MDD_%": rs["MDD_%"]})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- (c) ablation / feature-group importance ----------
def _abl_row(o):
    bt, pnl, ntr, _ = run_backtest(o)
    m = trading_metrics(bt["strat_ret"], bt["equity"], pnl, ntr, float(bt["position"].mean()))
    ml = ml_metrics(o["Actual_Next"].values, o["Pred_Next"].values, o["Close"].values)
    return {"Directional Accuracy (%)": ml["Directional Accuracy (%)"], "Sharpe": m["Sharpe"], "MDD (%)": m["MDD (%)"],
            "Total ROI (%)": m["Total ROI (%)"], "Avg Exposure (%)": m["Avg Exposure (%)"]}


def ablation_table(oos, ablated):
    """Same trained models, one feature GROUP neutralised (=train mean) or time-shuffled at test time. Thresholds/band unchanged."""
    rows = {"baseline (all features)": _abl_row(oos)}
    for k, arr in ablated.items():
        o = oos.copy(); o["W"] = arr[:, 0]; o["Pred_Ret"] = arr[:, 1]; o["Pred_Next"] = o["Close"].values * np.exp(arr[:, 1])
        rows[k] = _abl_row(o)
    df = pd.DataFrame(rows).T; b = df.iloc[0]
    df["dSharpe"] = df["Sharpe"] - b["Sharpe"]; df["dMDD (pp)"] = df["MDD (%)"] - b["MDD (%)"]; df["dDA (pp)"] = df["Directional Accuracy (%)"] - b["Directional Accuracy (%)"]
    return df


# ---------- scorecard: does it beat Buy & Hold? ----------
def beat_bh_scorecard(lstm_m, bh_m, rules_m, boot, exposure):
    sh_l, sh_b = lstm_m["Sharpe"], bh_m["Sharpe"]
    beats_sharpe = bool(sh_l > sh_b); sig = bool(boot["ci_lo"] > 0)
    beats_dd = bool(lstm_m["MDD (%)"] > bh_m["MDD (%)"] and lstm_m["Calmar"] > bh_m["Calmar"])
    best_name, best = max(rules_m.items(), key=lambda kv: kv[1]["Sharpe"])
    beats_rules = bool(sh_l > best["Sharpe"])
    if beats_sharpe and sig and beats_dd: label = "BEATS BUY & HOLD on risk-adjusted terms (statistically significant)"
    elif beats_sharpe and beats_dd: label = "BEATS BUY & HOLD in this sample, but NOT statistically distinguishable"
    elif beats_sharpe or beats_dd: label = "MIXED: better on one risk-adjusted measure only"
    else: label = "DOES NOT BEAT BUY & HOLD"
    f = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"
    txt = (f"SCORECARD: {label}. Sharpe {f(sh_l)} vs B&H {f(sh_b)} (difference {boot['point']:+.2f}, 95% block-bootstrap CI "
           f"[{boot['ci_lo']:+.2f}, {boot['ci_hi']:+.2f}], P(diff<=0)={boot['p_diff_le_0']:.2f}); MDD {f(lstm_m['MDD (%)'])}% vs {f(bh_m['MDD (%)'])}%; "
           f"Calmar {f(lstm_m['Calmar'])} vs {f(bh_m['Calmar'])}; CAGR {f(lstm_m['CAGR (%)'])}% vs {f(bh_m['CAGR (%)'])}% at {exposure*100:.0f}% average exposure "
           f"(a long-only strategy capped at 100% exposure will normally earn LESS total return than Buy & Hold; the question is risk-adjusted return). "
           f"Best simple rule: {best_name} (Sharpe {f(best['Sharpe'])}, MDD {f(best['MDD (%)'])}%) -> the LSTM "
           f"{'beats' if beats_rules else 'does NOT beat'} it" + ("" if beats_rules else ", so it adds nothing over that rule") + ".")
    return label, txt, dict(beats_sharpe=beats_sharpe, significant=sig, beats_dd=beats_dd, beats_rules=beats_rules)


# ---------- verdict: 200-day memory / crash protection ----------
def verdict(gate, crash, abl, lstm_m, rnn_m, hidden_pm, n_params, n_params_rnn, insample_end, exposure, flat_overall,
            med_epochs=None, em_m=None, skill=None):
    f = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"
    p = gate["pooled"]
    r_ratio, r_slope = p["forget_top vs Close/SMA200-1"]["pearson"], p["forget_top vs SMA200 slope(20d)"]["pearson"]
    pc_r, pc_s = gate["median_abs_pc1"]["Close/SMA200-1"], gate["median_abs_pc1"]["SMA200 slope(20d)"]
    ctl = gate["median_abs_pc1"].get("[control] Close/SMA50-1", float("nan"))
    crit_a = bool(np.nanmax(np.abs([r_ratio, r_slope])) >= 0.3 and np.nanmax([pc_r, pc_s]) >= 0.5)
    sc = crash.dropna(subset=["LSTM_MDD_%"]) if "LSTM_MDD_%" in crash else crash.iloc[0:0]
    degenerate = exposure < C.MIN_EXPOSURE
    ref = "EM_MDD_%" if "EM_MDD_%" in sc else "BH_MDD_%"
    if len(sc) == 0:
        crit_b, n_better, derisk = None, 0, False
    else:
        n_better = int((sc["LSTM_MDD_%"] > sc[ref]).sum())
        derisk = bool(sc["LSTM_avg_w_%"].mean() <= flat_overall - 10)          # average exposure lower in crashes than overall
        crit_b = bool(n_better / len(sc) >= 2 / 3 and derisk and not degenerate)
    tr = abl[abl.index.astype(str).str.startswith("trend200")]
    worst_da, worst_mdd, worst_sh = (tr["dDA (pp)"].min(), tr["dMDD (pp)"].min(), tr["dSharpe"].min()) if len(tr) else (np.nan,) * 3
    crit_c = bool(len(tr) and (worst_da <= -1.0 or worst_mdd <= -2.0 or worst_sh <= -0.1))
    n_true = int(crit_a) + int(bool(crit_b)) + int(crit_c)
    label = "INCONCLUSIVE" if (crit_b is None or degenerate) else ("SUPPORTED" if n_true == 3 else "NOT SUPPORTED" if n_true <= 1 else "INCONCLUSIVE")
    txt = (f"200-DAY MEMORY / CRASH-PROTECTION VERDICT: {label}. Average exposure {exposure*100:.1f}%. (a) Mean top-layer forget-gate correlates r={f(r_ratio)} with "
           f"Close/SMA200-1 and r={f(r_slope)} with the 20-day SMA_200 slope (pooled Pearson); median per-fold |r| of cell-state PC1 = {f(pc_r)} / {f(pc_s)} "
           f"(placebo 50-day trend: {f(ctl)}; a 200-day-specific memory should beat the placebo) -> {'met' if crit_a else 'not met'} (needs pooled |r|>=0.3 and PC1 |r|>=0.5). (b) ")
    if degenerate:
        txt += "average exposure is so low that crash protection is untestable. "
    elif crit_b is None:
        txt += "no crash window has enough out-of-sample days to score. "
    else:
        txt += (f"in {len(sc)} scored OOS crash window(s) the strategy had a shallower MDD than an exposure-matched constant-weight Buy & Hold in {n_better}; "
                f"mean return {f(sc['LSTM_ret_%'].mean())}% vs Buy & Hold {f(sc['BH_ret_%'].mean())}%; average exposure {f(sc['LSTM_avg_w_%'].mean())}% in crashes "
                f"vs {f(flat_overall)}% overall (needs >=10 pp lower in crashes) -> {'met' if crit_b else 'not met'}. ")
    txt += (f"(c) Neutralising/shuffling the 200-day feature group changes directional accuracy by up to {f(worst_da)} pp, MDD by up to {f(worst_mdd)} pp and Sharpe by "
            f"up to {f(worst_sh)} -> {'met' if crit_c else 'not met'} (needs DA drop >=1 pp, MDD worsening >=2 pp or Sharpe drop >=0.1). ")
    if skill:
        txt += (f"SKILL CHECK: corr(predicted, actual next-day return) = {skill['IC_pearson']:.3f}; corr(exposure, next-day return) = {skill['corr_exposure_vs_next_ret']:.3f}; "
                f"up-day base rate {skill['base_rate_up_days_%']:.1f}%. ")
    if rnn_m:
        txt += (f"Versus the Simple RNN: ROI {f(lstm_m['Total ROI (%)'])}% vs {f(rnn_m.get('Total ROI (%)'))}%, MDD {f(lstm_m['MDD (%)'])}% vs "
                f"{f(rnn_m.get('MDD (%)'))}%, Sharpe {f(lstm_m['Sharpe'])} vs {f(rnn_m.get('Sharpe'))}. ")
    txt += (f"CAVEATS: the initial training window runs to {insample_end} and contains the whole 2008 crash, so no 2008 claim is out-of-sample; the LSTM has "
            f"{n_params:,} parameters; gate/cell-state correlations between smooth trending series can be spurious and units are not comparable across retrained folds. ")
    if med_epochs is not None and med_epochs <= 2:
        txt += f"WARNING: median best epoch is {med_epochs:g} - the networks stop almost immediately, so (a) says little about learned memory. "
    txt += f"Heuristic thresholds; {n_true}/3 criteria met."
    return label, txt
