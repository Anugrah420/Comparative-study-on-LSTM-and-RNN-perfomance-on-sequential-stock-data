"""Expanding-window walk-forward (v5): position-sizing LSTM + auxiliary return head, seed ensemble.
Scalers are re-fit on each training window only; asserted below."""
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import config as C
from models import LSTMModel, extract_gates, count_params
from train import set_seed, train_model, predict, get_device


class WindowScaler:
    def fit(self, X, max_idx):
        self.mean, self.std, self.max_idx = X.mean(0), X.std(0) + 1e-8, int(max_idx); return self
    def transform(self, X): return (X - self.mean) / self.std
    def inverse(self, z): return z * self.std + self.mean


def make_windows(Xs, start, end):
    """Decision rows start..end-1 -> (n, WINDOW, F); window for row i = rows i-W+1..i (past only)."""
    seg = Xs[start - C.WINDOW + 1:end]
    return np.ascontiguousarray(sliding_window_view(seg, C.WINDOW, axis=0).transpose(0, 2, 1), dtype=np.float32)


def _alter(Xs, cols, mode, a0, end, seed):
    X2 = Xs.copy()
    if mode == "neutral":
        X2[:, cols] = 0.0                                                    # scaled 0 == training mean
    else:                                                                    # same time-shuffle for every column in the group
        perm = np.random.default_rng(seed).permutation(end - a0)
        X2[a0:end, cols] = Xs[a0:end][:, cols][perm]
    return X2


def run_walkforward(feat, hidden=None, collect_gates=False, ablate=False, tag="LSTM"):
    hidden = hidden or C.HIDDEN
    feats = list(C.FEATURES)
    X = feat[feats].values.astype(np.float64)
    y = feat["Target_Ret"].values.astype(np.float64)                          # next-day log return
    r = np.expm1(y)                                                           # simple return (utility)
    close, ynext = feat["Close"].values, feat["Target_Close"].values          # price levels: evaluation only
    groups = {g: [feats.index(f) for f in fs] for g, fs in C.FEATURE_GROUPS.items()}
    N = len(feat); device = get_device()
    T = C.INITIAL_TRAIN_YEARS * C.TRADING_DAYS
    rows, folds, gates, abl = [], [], [], defaultdict(list)
    fold = 0; t_start = time.time(); n_folds = int(np.ceil((N - T) / C.RETRAIN_EVERY))
    while T < N:
        end = min(T + C.RETRAIN_EVERY, N)
        if N - end < C.MIN_LAST_FOLD: end = N                                 # avoid retraining for a tiny final block
        tr_rows = np.arange(C.WINDOW - 1, T); n_val = int(round(C.VAL_FRAC * len(tr_rows)))
        fit_rows, val_rows = tr_rows[:-n_val], tr_rows[-n_val:]
        xs = WindowScaler().fit(X[:fit_rows[-1] + 1], fit_rows[-1])           # only rows <= last fit row
        ys = WindowScaler().fit(y[fit_rows].reshape(-1, 1), fit_rows[-1])
        assert xs.max_idx < T and ys.max_idx < T and val_rows[-1] < T, "scaler/indicator used data beyond training window"
        assert feat["Next_Date"].values[T - 1] <= feat.index.values[T], "training label realised after OOS start"
        Xs = xs.transform(X[:end]).astype(np.float32)                           # rows < end only
        Xtr, Xva = make_windows(Xs, fit_rows[0], fit_rows[-1] + 1), make_windows(Xs, val_rows[0], val_rows[-1] + 1)
        ytr, yva = ys.transform(y[fit_rows].reshape(-1, 1)).ravel(), ys.transform(y[val_rows].reshape(-1, 1)).ravel()
        rn = r / r[fit_rows].std()                                              # returns in units of training volatility
        rtr, rva = rn[fit_rows], rn[val_rows]
        lam = max(C.RISK_KAPPA * rtr.mean() / np.mean(rtr ** 2), 0.01)          # unconditional optimal exposure = 1/RISK_KAPPA
        Xo = make_windows(Xs, T, end)
        inv = lambda p: ys.inverse(p.reshape(-1, 1)).ravel()
        alts = {}
        if ablate:
            a0 = T - C.WINDOW + 1
            for gname, cols in groups.items():
                for mode in ("neutral", "shuffle"):
                    alts[f"{gname}:{mode}"] = make_windows(_alter(Xs, cols, mode, a0, end, C.SEED + fold), T, end)
        ws, rets, infos, pa = [], [], [], defaultdict(list)
        for s in range(C.N_SEEDS):
            set_seed(C.SEED + s)
            model = LSTMModel(X.shape[1], hidden)
            model, info = train_model(model, Xtr, ytr, rtr, Xva, yva, rva, device, seed=C.SEED + 1000 * s + fold, lam=lam)
            print(f"    [{tag}] fold {fold} seed {s+1}/{C.N_SEEDS}: stopped at epoch {info['epochs_run']}, best epoch {info['best_epoch']} "
                  f"({time.time()-t_start:.0f}s elapsed)", flush=True)
            infos.append(info)
            p = predict(model, Xo, device); ws.append(p[:, 0]); rets.append(inv(p[:, 1]))
            if collect_gates and s == 0:
                gates.append(extract_gates(model, Xo, device))                  # gates from the seed-0 model only
            for k, XX in alts.items():
                q = predict(model, XX, device); pa[k].append(np.column_stack([q[:, 0], inv(q[:, 1])]))
        w_hat, ret_hat = np.mean(ws, 0), np.mean(rets, 0)
        rows.append(pd.DataFrame({"Close": close[T:end], "W": w_hat, "Pred_Ret": ret_hat, "Pred_Next": close[T:end] * np.exp(ret_hat),
                                  "Actual_Next": ynext[T:end], "Next_Date": feat["Next_Date"].values[T:end], "Fold": fold},
                                 index=feat.index[T:end]))
        for k, v in pa.items(): abl[k].append(np.mean(v, 0))
        folds.append(dict(fold=fold, train_end=str(feat.index[T - 1].date()), oos_start=str(feat.index[T].date()),
                          oos_end=str(feat.index[end - 1].date()), n_train=int(len(fit_rows)), n_val=int(len(val_rows)),
                          lam=float(lam), mean_exposure=float(w_hat.mean()), best_epochs=[i["best_epoch"] for i in infos],
                          val_loss=[round(i["best_val_loss"], 4) for i in infos]))
        print(f"[{tag}] fold {fold:2d} train<= {feat.index[T-1].date()} OOS {feat.index[T].date()}..{feat.index[end-1].date()} "
              f"mean_w={w_hat.mean():.2f} (min {w_hat.min():.2f}, max {w_hat.max():.2f}) epochs={folds[-1]['best_epochs']} | "
              f"{fold+1}/~{n_folds} folds, {(time.time()-t_start)/60:.1f} min elapsed", flush=True)
        T, fold = end, fold + 1
    oos = pd.concat(rows)
    g = {k: np.concatenate([p[k] for p in gates]) for k in gates[0]} if gates else None
    a = {k: np.concatenate(v) for k, v in abl.items()} if ablate else None   # (n,2) [exposure, predicted return] per variant
    return dict(oos=oos, folds=folds, gates=g, ablation=a, n_params=count_params(LSTMModel(X.shape[1], hidden)), hidden=hidden)
