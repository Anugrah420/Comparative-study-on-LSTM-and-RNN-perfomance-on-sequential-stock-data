"""
main.py — LSTM on INTRADAY AAPL (5m). Counterpart of the Simple RNN intraday
experiment; everything is identical except the recurrent cell.

Usage
    python main.py                          # download fresh 60d data
    python main.py --data-csv path.csv      # reuse the SAME raw snapshot as the RNN run
    python main.py --param-matched          # + capacity-matched LSTM robustness run
    python main.py --skip-ablation          # faster
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import config as C
import diagnostics as D
import metrics as M
import plots as P
from backtest import buy_and_hold, generate_signals, run_backtest
from data import prepare, to_price
from models import LSTMRegressor, count_params, extract_gates, param_matched_hidden, print_param_report
from train import get_device, predict, set_seed, train_model

FX_NOTE = ("FX assumption: AAPL trades in USD. Strategy returns are compounded with fractional "
           "shares and equity is expressed in ₹ as INITIAL_CAPITAL × cumulative growth factor, "
           "i.e. USD/INR is held constant over the test window.")

LIMITATIONS = """LIMITATIONS
 • Sample size: Yahoo caps 5-minute history at ~60 days, so the test set is only ~9 trading days;
   every metric (Sharpe, PF, gate correlations) has wide error bars. Results also change with the
   download date — reuse the cached raw CSV to pair the RNN and LSTM runs exactly.
 • Single asset (AAPL), single regime; no evidence this generalises to other stocks or periods.
 • Overfitting risk: ~1–1.5k training windows vs ~50k LSTM parameters; early stopping on a
   ~9-day validation set is itself noisy. A single seed is reported.
 • Target is the next-bar log return (converted back to a price for RMSE/MAPE). Price-level RMSE
   is dominated by the current price, so always compare against the persistence baseline.
 • Costs: 0.05 % per side, no slippage, no borrow fees, fills at bar close. Results are highly
   sensitive to this; re-run with COST_PER_SIDE = 0.0010 to stress-test.
 • FX held constant (see note); fractional shares and 100 % equity per trade are idealisations.
 • Confound: the LSTM has ~4x the RNN's recurrent parameters; gate statistics are correlational."""


def load_rnn_artifacts(rnn_dir):
    rnn_json = M.load_json(rnn_dir / C.RNN_RESULTS_JSON_NAME)
    rnn_pred = None
    csv = rnn_dir / C.RNN_PREDICTIONS_CSV_NAME
    if csv.exists():
        df = pd.read_csv(csv)
        if {"target_timestamp", "pred_next"} <= set(df.columns):
            idx = pd.to_datetime(df["target_timestamp"], utc=True).dt.tz_convert(C.MARKET_TZ)
            rnn_pred = pd.Series(df["pred_next"].to_numpy(float), index=idx, name="rnn_pred")
    print(f"[rnn] results JSON: {'found' if rnn_json else 'not found'} | "
          f"predictions CSV: {'found' if rnn_pred is not None else 'not found'}  ({rnn_dir})")
    return rnn_json, rnn_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", default=None, help="raw OHLCV snapshot to reuse")
    ap.add_argument("--param-matched", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--rnn-dir", default=str(C.RNN_OUT_DIR))
    args = ap.parse_args()

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(C.SEED)
    device = get_device()
    print(f"[env] device={device} torch={torch.__version__}")

    # 1 ── data ──────────────────────────────────────────────────────────
    B = prepare(args.data_csv)
    feat, splits, scalers, ws, test_bars = B["feat"], B["splits"], B["scalers"], B["windows"], B["test_bars"]
    n_feat = len(C.FEATURES)

    # 2 ── model ─────────────────────────────────────────────────────────
    set_seed(C.SEED)
    model = LSTMRegressor(n_feat)
    params = print_param_report(model, n_feat)

    # 3 ── train ─────────────────────────────────────────────────────────
    model, hist = train_model(model, ws["train"], ws["val"], device, tag="LSTM")
    torch.save(model.state_dict(), C.WEIGHTS_PT)
    hist.to_csv(C.OUT_DIR / "training_history_intraday_lstm.csv", index=False)
    P.plot_training_curve(hist, C.FIG_DIR / "training_curve_lstm.png")

    # 4 ── predict on test ───────────────────────────────────────────────
    meta = ws["test"].meta.copy()
    close_t = meta["close_t"].to_numpy()
    pred = to_price(scalers, predict(model, ws["test"].X, device), close_t)
    actual = meta["actual_next"].to_numpy()

    # 5 ── gates ─────────────────────────────────────────────────────────
    gates = extract_gates(model, ws["test"].X, device)
    assert np.allclose(to_price(scalers, gates["pred_scaled"], close_t), pred, atol=1e-3)
    np.savez_compressed(C.GATES_NPZ, forget=gates["forget"], input=gates["input"],
                        timestamp=meta["timestamp"].astype(str).to_numpy(),
                        layout=np.array("[sample, layer, timestep, hidden_unit]"))
    print(f"[save] gate activations -> {C.GATES_NPZ}  shape={gates['forget'].shape}")

    # 6 ── backtests ─────────────────────────────────────────────────────
    sig = generate_signals(pred, close_t)
    sig_series = pd.Series(sig, index=pd.DatetimeIndex(meta["timestamp"]))
    bt_lstm = run_backtest(test_bars, sig_series)
    bt_bh = buy_and_hold(test_bars)
    persist_pred = close_t.copy()                       # random-walk forecast
    bt_pers = run_backtest(test_bars, pd.Series(generate_signals(persist_pred, close_t),
                                                index=sig_series.index))

    rows = {
        "LSTM": M.evaluate(bt_lstm, close_t, actual, pred),
        "Buy & Hold": M.evaluate(bt_bh),
        "Persistence": M.evaluate(bt_pers, close_t, actual, persist_pred),
    }

    # 6b ── optional parameter-matched robustness run ────────────────────
    if args.param_matched:
        h = param_matched_hidden(n_feat)
        set_seed(C.SEED)
        pm = LSTMRegressor(n_feat, hidden=h)
        print(f"[robustness] param-matched LSTM: hidden={h}, recurrent params="
              f"{count_params(pm)['recurrent']:,} (RNN: {params['rnn_reference']['recurrent']:,})")
        pm, _ = train_model(pm, ws["train"], ws["val"], device, tag=f"LSTM-h{h}")
        p_pm = to_price(scalers, predict(pm, ws["test"].X, device), close_t)
        bt_pm = run_backtest(test_bars, pd.Series(generate_signals(p_pm, close_t), index=sig_series.index))
        rows[f"LSTM (param-matched, h={h}) [robustness]"] = M.evaluate(bt_pm, close_t, actual, p_pm)
        params["param_matched_hidden"] = h

    matrix = M.build_matrix(rows)
    print("\n══════════ EVALUATION MATRIX — TEST SET (AAPL 5m) ══════════")
    print(M.pretty_matrix(matrix).to_string())
    print("Persistence never clears the cost threshold, so it trades 0 times (DA undefined).")
    print(FX_NOTE)

    # 7 ── RNN comparison ────────────────────────────────────────────────
    from pathlib import Path
    rnn_json, rnn_pred = load_rnn_artifacts(Path(args.rnn_dir))
    cmp = M.compare_with_rnn(rows["LSTM"], rnn_json)
    if cmp is not None:
        print("\n══════════ SIMPLE RNN vs LSTM (test set) ══════════")
        print(cmp.to_string(float_format=lambda v: f"{v:,.4f}"))
        cmp.to_csv(C.OUT_DIR / "rnn_vs_lstm_intraday.csv")

    # 8 ── diagnostics ───────────────────────────────────────────────────
    vol_z = D.volume_zscore(feat, splits["train"])
    # 75th pct over TRAIN decision bars (same time-of-day mix as test samples)
    vol_thr = float(np.nanpercentile(ws["train"].meta["ret_std_12_t"], C.VOL_PCTL))
    diag = {
        "gates": D.gate_volume_analysis(gates, meta, vol_z),
        "vwap": D.vwap_analysis(meta, pred, rnn_pred),
        "volatility": D.volatility_analysis(meta, pred, vol_thr),
    }
    if args.skip_ablation:
        diag["ablation"] = {"shuffle": {"delta_rmse_pct_mean": float("nan"), "delta_da_pp_mean": float("nan")}}
    else:
        diag["ablation"] = D.volume_ablation(model, feat, scalers, splits["test"], meta, pred, device)
    print("\n══════════ DIAGNOSTICS ══════════")
    for k, v in diag.items():
        print(f"[{k}]")
        for kk, vv in v.items():
            print(f"   {kk}: {vv}")

    # 9 ── predictions CSV ───────────────────────────────────────────────
    forget_bar = D.forget_mean_by_bar(gates, meta)
    out_csv = meta.drop(columns=["row_end"]).copy()
    out_csv["pred_next"] = pred
    out_csv["pred_change"] = pred - close_t
    out_csv["signal"] = sig
    out_csv["high_vol"] = meta["ret_std_12_t"].to_numpy() > vol_thr
    out_csv["volume_z_t"] = vol_z[meta["row_end"].to_numpy()]
    out_csv["forget_mean_t"] = forget_bar.to_numpy()
    out_csv["input_mean_t"] = gates["input"].astype(np.float32)[:, :, -1, :].mean(axis=(1, 2))
    out_csv.to_csv(C.PREDICTIONS_CSV, index=False)
    print(f"[save] predictions -> {C.PREDICTIONS_CSV}")

    # 10 ── plots ────────────────────────────────────────────────────────
    pred_by_target = pd.Series(pred, index=pd.DatetimeIndex(meta["target_timestamp"]))
    P.plot_three_day(test_bars, pred_by_target, forget_bar, vol_thr,
                     C.FIG_DIR / "three_day_lstm.png", C.FIG_DIR / "three_day_lstm.html", rnn_pred)
    P.plot_equity({"LSTM": bt_lstm["equity"], "Buy & Hold": bt_bh["equity"], "Persistence": bt_pers["equity"]},
                  C.FIG_DIR / "equity_lstm.png", C.FIG_DIR / "equity_lstm.html")
    P.plot_gate_vs_volume(forget_bar.to_numpy(), vol_z[meta["row_end"].to_numpy()],
                          C.FIG_DIR / "forget_gate_vs_volume.png")
    print(f"[save] figures -> {C.FIG_DIR}")

    # 11 ── verdict + JSON ───────────────────────────────────────────────
    label, text = D.verdict(diag, (rnn_json or {}).get("diagnostics"), params["recurrent_ratio"],
                            len(splits["test"]))
    print("\n" + text)

    metrics_block = {"Strategy": rows["LSTM"], "Buy & Hold": rows["Buy & Hold"], "Persistence": rows["Persistence"]}
    for k in rows:
        if k.startswith("LSTM (param-matched"):
            metrics_block[k] = rows[k]
    M.save_results(C.RESULTS_JSON, {
        "experiment": C.EXPERIMENT, "model": "LSTM", "ticker": C.TICKER,
        "config": {k: getattr(C, k) for k in ["SEED", "WINDOW", "HIDDEN", "NUM_LAYERS", "DROPOUT", "BATCH",
                                              "LR", "EPOCHS", "PATIENCE", "GRAD_CLIP", "COST_PER_SIDE",
                                              "INITIAL_CAPITAL", "ANN_FACTOR", "RF_ANNUAL", "FEATURES"]},
        "data": {"test_days": [str(d) for d in splits["test"]], "n_test_samples": int(len(meta))},
        "params": params, "metrics": metrics_block, "diagnostics": diag,
        "verdict_label": label, "verdict": text, "fx_note": FX_NOTE,
    })
    print("\n" + LIMITATIONS)


if __name__ == "__main__":
    main()
