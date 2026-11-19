"""
diagnostics.py — tests of the hypothesis:
  "The LSTM's forget gate filters out 5-minute volume noise while tracking
   VWAP momentum."

(a) forget-gate vs volume z-score / volume spikes
(b) predicted 1-bar change vs (VWAP − Close)/VWAP  (and RNN if available)
(c) volatility-conditioned diagnostics (identical to the RNN experiment)
(d) Volume ablation at test time (shuffle + multiplicative noise)
+ a rule-based verdict paragraph.

(c) and (d) are model-agnostic and are the same code the RNN experiment uses;
(a) is LSTM-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from data import make_windows, to_price
from metrics import directional_accuracy, regression_metrics
from train import predict


# ── helpers ──────────────────────────────────────────────────────────────
def _corr(a, b, method="pearson"):
    s = pd.DataFrame({"a": np.asarray(a, float), "b": np.asarray(b, float)}).dropna()
    if len(s) < 3 or s["a"].std() == 0 or s["b"].std() == 0:
        return float("nan")
    return float(s["a"].corr(s["b"], method=method))


def _perm_test(x, mask, n=C.N_PERMUTATIONS, seed=C.SEED):
    """Two-sided permutation test of mean(x[mask]) − mean(x[~mask])."""
    x, mask = np.asarray(x, float), np.asarray(mask, bool)
    if mask.sum() < 5 or (~mask).sum() < 5:
        return float("nan"), float("nan")
    obs = x[mask].mean() - x[~mask].mean()
    rng = np.random.default_rng(seed)
    k = mask.sum()
    null = np.empty(n)
    for j in range(n):
        p = rng.permutation(len(x))
        null[j] = x[p[:k]].mean() - x[p[k:]].mean()
    return float(obs), float((np.abs(null) >= abs(obs)).mean())


def volume_zscore(feat: pd.DataFrame, train_days) -> np.ndarray:
    """Time-of-day-adjusted log-volume z-score (removes the intraday U-shape).
    Slot statistics come from TRAIN rows only."""
    lv = np.log1p(feat["Volume"])
    tr = feat["session"].isin(train_days)
    slot = feat["minute_of_day"]
    med = lv[tr].groupby(slot[tr]).median()
    sd = lv[tr].groupby(slot[tr]).std()
    sd = sd.where(sd > 0, lv[tr].std())
    z = (lv - slot.map(med)) / slot.map(sd)
    return z.to_numpy(float)


# ── (a) gates vs volume ──────────────────────────────────────────────────
def gate_volume_analysis(gates: dict, meta: pd.DataFrame, vol_z: np.ndarray) -> dict:
    W = C.WINDOW
    f = gates["forget"].astype(np.float32)          # [N, L, T, H]
    ig = gates["input"].astype(np.float32)
    f_bar, i_bar = f.mean(axis=(1, 3)), ig.mean(axis=(1, 3))        # [N, T]
    rows = meta["row_end"].to_numpy()[:, None] - (W - 1) + np.arange(W)[None, :]
    z = vol_z[rows]                                                  # [N, T]

    # Primary: last timestep only -> one observation per bar (no window overlap)
    f_last, i_last, z_last = f_bar[:, -1], i_bar[:, -1], z[:, -1]
    spike = z_last > C.SPIKE_Z
    f_diff, f_p = _perm_test(f_last, spike)
    i_diff, i_p = _perm_test(i_last, spike)

    # Step AFTER a spike: does the cell erase what it just wrote?
    f_next, spike_prev = f_bar[:, 1:].ravel(), (z[:, :-1] > C.SPIKE_Z).ravel()

    res = {
        "n_bars": int(len(f_last)), "n_spikes": int(spike.sum()), "spike_z": C.SPIKE_Z,
        "pearson_forget_vs_volz": _corr(f_last, z_last),
        "spearman_forget_vs_volz": _corr(f_last, z_last, "spearman"),
        "forget_mean_spike": float(f_last[spike].mean()) if spike.any() else float("nan"),
        "forget_mean_nonspike": float(f_last[~spike].mean()),
        "forget_diff_spike_minus_non": f_diff, "forget_perm_p": f_p,
        "pearson_input_vs_volz": _corr(i_last, z_last),
        "input_diff_spike_minus_non": i_diff, "input_perm_p": i_p,
        "forget_next_step_diff_after_spike": float(f_next[spike_prev].mean() - f_next[~spike_prev].mean())
        if spike_prev.any() else float("nan"),
        # all (sample, timestep) pairs — overlapping windows, descriptive only
        "pearson_forget_vs_volz_all_steps": _corr(f_bar.ravel(), z.ravel()),
        "per_layer_pearson_forget_vs_volz": [
            _corr(f[:, k, -1, :].mean(-1), z_last) for k in range(f.shape[1])
        ],
    }
    return res


def forget_mean_by_bar(gates: dict, meta: pd.DataFrame) -> pd.Series:
    """Mean forget gate (layers x units) at the last timestep, i.e. while
    processing bar t, indexed by bar-t timestamp."""
    f = gates["forget"].astype(np.float32)[:, :, -1, :].mean(axis=(1, 2))
    return pd.Series(f, index=pd.DatetimeIndex(meta["timestamp"]), name="forget_mean")


# ── (b) VWAP momentum ────────────────────────────────────────────────────
def vwap_analysis(meta: pd.DataFrame, pred_next: np.ndarray, rnn_pred: pd.Series | None) -> dict:
    dev = (meta["vwap_t"] - meta["close_t"]) / meta["vwap_t"]   # >0: price below VWAP
    pred_chg = pred_next - meta["close_t"].to_numpy()
    act_chg = (meta["actual_next"] - meta["close_t"]).to_numpy()
    res = {
        "corr_pred_change_vs_vwap_dev": _corr(pred_chg, dev),
        "spearman_pred_change_vs_vwap_dev": _corr(pred_chg, dev, "spearman"),
        "corr_actual_change_vs_vwap_dev": _corr(act_chg, dev),
        "interpretation": "positive = predicts reversion toward VWAP; negative = predicts drift away",
        "rnn_corr_pred_change_vs_vwap_dev": float("nan"),
    }
    if rnn_pred is not None:
        r = rnn_pred.reindex(pd.DatetimeIndex(meta["target_timestamp"])).to_numpy(float)
        res["rnn_corr_pred_change_vs_vwap_dev"] = _corr(r - meta["close_t"].to_numpy(), dev)
    return res


# ── (c) volatility-conditioned diagnostics (same as RNN experiment) ─────
def volatility_analysis(meta: pd.DataFrame, pred_next: np.ndarray, vol_threshold: float) -> dict:
    close = meta["close_t"].to_numpy()
    act = meta["actual_next"].to_numpy()
    pred_chg, act_chg = pred_next - close, act - close
    high = meta["ret_std_12_t"].to_numpy() > vol_threshold
    out = {"vol_threshold_ret_std_12": vol_threshold}
    for name, m in [("high_vol", high), ("low_vol", ~high)]:
        if m.sum() < 3:
            out[name] = {"n": int(m.sum())}
            continue
        rmse, _ = regression_metrics(act[m], pred_next[m])
        out[name] = {
            "n": int(m.sum()),
            "std_ratio_pred_to_actual": float(np.std(pred_chg[m]) / np.std(act_chg[m]))
            if np.std(act_chg[m]) > 0 else float("nan"),
            "rmse_usd": rmse,
            "directional_accuracy_pct": directional_accuracy(close[m], act[m], pred_next[m]),
        }
    sess = meta["session"].to_numpy()
    flips_p, flips_a = [], []
    for d in np.unique(sess):
        m = sess == d
        sp, sa = np.sign(pred_chg[m]), np.sign(act_chg[m])
        flips_p.append(int((np.diff(sp[sp != 0]) != 0).sum()))
        flips_a.append(int((np.diff(sa[sa != 0]) != 0).sum()))
    out["flips_per_day_pred"] = float(np.mean(flips_p))
    out["flips_per_day_actual"] = float(np.mean(flips_a))
    return out


# ── (d) Volume ablation ─────────────────────────────────────────────────
def volume_ablation(model, feat, scalers, test_days, base_meta, base_pred, device,
                    repeats: int = C.ABLATION_REPEATS) -> dict:
    """Perturb ONLY the Volume input column on test rows, re-window, re-predict.
    VWAP (already computed from true volume) is untouched — this isolates the
    model's use of the raw Volume input."""
    close, act = base_meta["close_t"].to_numpy(), base_meta["actual_next"].to_numpy()
    rmse0, _ = regression_metrics(act, base_pred)
    da0 = directional_accuracy(close, act, base_pred)
    test_mask = feat["session"].isin(test_days).to_numpy()
    out = {"base_rmse": rmse0, "base_da": da0}

    for mode in ["shuffle", "lognormal_noise"]:
        d_rmse, d_da = [], []
        for k in range(repeats):
            rng = np.random.default_rng(C.SEED + k)
            f2 = feat.copy()
            v = f2.loc[test_mask, "Volume"].to_numpy()
            f2.loc[test_mask, "Volume"] = (rng.permutation(v) if mode == "shuffle"
                                           else v * np.exp(rng.normal(0, 0.5, len(v))))
            ws = make_windows(f2, scalers, test_days)
            assert (ws.meta["target_timestamp"].values == base_meta["target_timestamp"].values).all()
            p = to_price(scalers, predict(model, ws.X, device), close)
            r, _ = regression_metrics(act, p)
            d_rmse.append((r - rmse0) / rmse0 * 100)
            d_da.append(directional_accuracy(close, act, p) - da0)
        out[mode] = {"delta_rmse_pct_mean": float(np.mean(d_rmse)), "delta_rmse_pct_std": float(np.std(d_rmse)),
                     "delta_da_pp_mean": float(np.mean(d_da)), "delta_da_pp_std": float(np.std(d_da))}
    return out


# ── Verdict ─────────────────────────────────────────────────────────────
def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return float("nan")
        d = d[k]
    return float(d)


def verdict(diag: dict, rnn_diag: dict | None, ratio: float, n_test_days: int) -> tuple[str, str]:
    g, vw, vol, ab = diag["gates"], diag["vwap"], diag["volatility"], diag["ablation"]
    rnn_diag = rnn_diag or {}

    # (a) forget gate lower on spikes (significant) => "forgets" spikes
    a_diff, a_p = g["forget_diff_spike_minus_non"], g["forget_perm_p"]
    A = bool(np.isfinite(a_diff) and a_diff < 0 and a_p < 0.05)
    A_contra = bool(np.isfinite(a_diff) and a_diff > 0 and a_p < 0.05)

    # (b) VWAP tracking: meaningful corr with the realised sign, and > RNN if known
    c_l, c_a = vw["corr_pred_change_vs_vwap_dev"], vw["corr_actual_change_vs_vwap_dev"]
    c_r = vw.get("rnn_corr_pred_change_vs_vwap_dev", float("nan"))
    if not np.isfinite(c_r):
        c_r = _get(rnn_diag, "vwap", "corr_pred_change_vs_vwap_dev")
    B = bool(np.isfinite(c_l) and abs(c_l) >= 0.10 and np.sign(c_l) == np.sign(c_a)
             and (not np.isfinite(c_r) or abs(c_l) > abs(c_r)))

    # (c) smoother in high vol: std ratio < 1 and no more flips than the RNN
    sr = _get(vol, "high_vol", "std_ratio_pred_to_actual")
    fl, fl_r = vol["flips_per_day_pred"], _get(rnn_diag, "volatility", "flips_per_day_pred")
    Cc = bool(np.isfinite(sr) and sr < 1 and (not np.isfinite(fl_r) or fl <= fl_r))

    # (d) low Volume sensitivity (< 2 % RMSE change) and below the RNN's if known
    d_l = abs(ab["shuffle"]["delta_rmse_pct_mean"])
    d_r = abs(_get(rnn_diag, "ablation", "shuffle", "delta_rmse_pct_mean"))
    D = bool(d_l < 2.0 and (not np.isfinite(d_r) or d_l < d_r))

    passed = sum([B, Cc, D])
    if A and passed >= 2:
        label = "SUPPORTED"
    elif A_contra or (not A and passed <= 1):
        label = "NOT SUPPORTED"
    else:
        label = "INCONCLUSIVE"

    rnn_txt = "" if not np.isfinite(c_r) else f" (RNN: {c_r:+.3f})"
    abl_rnn = "" if not np.isfinite(d_r) else f" vs {d_r:.2f}% for the RNN"
    flip_rnn = "" if not np.isfinite(fl_r) else f" vs {fl_r:.1f} for the RNN"
    text = (
        f"VERDICT: {label}. (a) Forget gate on volume spikes (z>{C.SPIKE_Z:g}, n={g['n_spikes']}): "
        f"mean {g['forget_mean_spike']:.3f} vs {g['forget_mean_nonspike']:.3f} otherwise "
        f"(Δ={a_diff:+.4f}, permutation p={a_p:.3f}); Pearson r(forget, vol-z)={g['pearson_forget_vs_volz']:+.3f}"
        f" -> {'gate drops on spikes' if A else 'no reliable drop on spikes'}. "
        f"(b) r(predicted Δ, (VWAP−Close)/VWAP)={c_l:+.3f}{rnn_txt} vs realised {c_a:+.3f} -> "
        f"{'tracks VWAP' if B else 'VWAP tracking not established'}. "
        f"(c) High-vol std(predΔ)/std(actualΔ)={sr:.3f}; direction flips/day={fl:.1f}{flip_rnn}. "
        f"(d) Shuffling Volume changes RMSE by {ab['shuffle']['delta_rmse_pct_mean']:+.2f}% and DA by "
        f"{ab['shuffle']['delta_da_pp_mean']:+.2f} pp{abl_rnn} -> "
        f"{'low sensitivity (consistent with filtering)' if D else 'not clearly filtered'}. "
        f"Confounds: the LSTM has {ratio:.1f}x the RNN's recurrent parameters, so differences may reflect "
        f"capacity rather than gating (see the param-matched row); the test set is only {n_test_days} days of a "
        f"60-day sample, so these statistics are noisy and gate/volume correlation is not causation."
    )
    return label, text
