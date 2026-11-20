"""LONG-TERM DAILY experiment v5: feature-rich LSTM that SIZES a long-only position (defaults to Buy & Hold).
Run:   python main.py                    (needs internet for yfinance; falls back to cached files in the output folder)
       python main.py --device cuda      (force GPU; errors clearly if none. Default --device auto = GPU if available)
       python main.py --seeds 1          (faster: single seed instead of the 3-seed ensemble)
       python main.py --no-exog          (price/volume features only; skip VIX/rates/credit/gold/bonds)
       python main.py --smoke            (OFFLINE plumbing test on SYNTHETIC data, 2 epochs; NOT real data, NOT a result)"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import config as C
import data, metrics as M, interpret as I
from backtest import run_backtest, rule_benchmarks


def load_rnn():
    mp = [p for p in (os.path.join(C.OUT_DIR, C.RNN_RESULTS), C.RNN_RESULTS) if os.path.exists(p)]
    ep = [p for p in (os.path.join(C.OUT_DIR, C.RNN_EQUITY), C.RNN_EQUITY) if os.path.exists(p)]
    res = json.load(open(mp[0])) if mp else None
    eq = pd.read_csv(ep[0], parse_dates=["date"]).set_index("date") if ep else None
    return res, eq


def main(smoke=False):
    if smoke:
        C.EPOCHS, C.OUT_DIR = 2, "outputs_smoke"
    os.makedirs(C.OUT_DIR, exist_ok=True); P = lambda n: os.path.join(C.OUT_DIR, n)
    from train import set_seed, get_device
    import walkforward as W, models, plots
    set_seed(C.SEED); get_device()
    print("ASSET:", C.TICKER, "-", C.TICKERS_NOTE)
    print({k: getattr(C, k) for k in ("SEED", "WINDOW", "HIDDEN", "NUM_LAYERS", "DROPOUT", "BATCH", "LR", "EPOCHS", "PATIENCE", "N_SEEDS",
                                       "RISK_KAPPA", "REBAL_BAND", "AUX_WEIGHT", "INITIAL_CAPITAL")})
    print(f"ASSUMPTION: {C.TICKER} trades in USD; Rs equity = INITIAL_CAPITAL x cumulative growth factor (fractional shares, no FX conversion/FX risk).")

    # ---------- data ----------
    raw = data.make_synthetic() if smoke else data.download_prices(P(f"{C.safe_name(C.TICKER)}_raw.csv"))
    data.verify_coverage(raw)
    exog = data.load_exog(raw.index, C.OUT_DIR) if (C.USE_EXOG and not smoke) else None
    feat = data.build_features(raw, exog); data.assert_no_lookahead(raw, exog)
    print(f"[features] {len(C.FEATURES)} inputs; groups: " + "; ".join(f"{g}={len(v)}" for g, v in C.FEATURE_GROUPS.items()))
    T0 = C.INITIAL_TRAIN_YEARS * C.TRADING_DAYS
    ins_start, ins_end, oos_start = feat.index[0], feat.index[T0 - 1], feat.index[T0]
    print(f"[split] post-warm-up data {ins_start.date()}..{feat.index[-1].date()} ({len(feat)} rows). Initial training window "
          f"{ins_start.date()}..{ins_end.date()} (in-sample, shaded grey); first OOS decision {oos_start.date()}. "
          f"NOTE: this window contains the 2008 crash -> any 2008 result is in-sample.")
    nf = len(C.FEATURES); n_lstm = models.count_params(models.LSTMModel(nf)); n_rnn = models.count_params(models.RNNReference(nf))
    h_pm, _ = models.match_hidden(nf)
    print(f"[params] LSTM(hidden={C.HIDDEN}) = {n_lstm:,} parameters vs a same-size Simple RNN = {n_rnn:,} ({n_lstm/n_rnn:.2f}x).")

    # ---------- walk-forward ----------
    wf = W.run_walkforward(feat, collect_gates=True, ablate=True, tag="LSTM")
    oos = wf["oos"]; bt, pnl, ntr, bh_pnl = run_backtest(oos)
    raw_m = M.full_matrix(oos, bt, pnl, ntr, bh_pnl); lstm_m, bh_m = raw_m["strategy"], raw_m["buy_hold"]
    exposure = float(bt["position"].mean())
    em_m = M.exposure_matched(bt, exposure); skill = M.skill_diagnostics(oos, bt)
    rules = rule_benchmarks(feat, oos); rules_m = M.rule_metrics(rules)
    boot = M.sharpe_diff_bootstrap(bt["strat_ret"].values, bt["bh_ret"].values)
    all_epochs = [e for fo in wf["folds"] for e in fo["best_epochs"]]; med_epochs = float(np.median(all_epochs))

    cols = {"LSTM Strategy": lstm_m, "Buy & Hold": bh_m, f"B&H x {exposure*100:.0f}% (const.)": em_m, **rules_m, "Persistence": raw_m["persistence"]}
    print("\n=== EVALUATION MATRIX (OUT-OF-SAMPLE ONLY; every strategy pays 0.05%/side on traded notional) ===\n" + M.matrix_table(cols).to_string())
    sc_label, sc_text, sc_flags = I.beat_bh_scorecard(lstm_m, bh_m, rules_m, boot, exposure)
    print("\n" + sc_text)
    print("\n=== SKILL DIAGNOSTICS (does the model predict anything?) ===")
    print("  " + "\n  ".join(f"{k}: {v:.3f}" for k, v in skill.items()))
    print(f"  median best epoch across folds/seeds: {med_epochs:g}" + ("  <-- WARNING: nets stop almost immediately (barely trained)" if med_epochs <= 2 else ""))

    rnn_res, rnn_eq = load_rnn(); rnn_m = None
    if rnn_res:
        rnn_m = rnn_res.get("metrics", {}).get("strategy", {})
        rows = {k: [rnn_m.get(k, np.nan), lstm_m.get(k, np.nan), (lstm_m.get(k, np.nan) - rnn_m[k]) if rnn_m.get(k) is not None else np.nan] for k in M.ORDER}
        print("\n=== SIDE-BY-SIDE: Simple RNN vs LSTM ===\n" + pd.DataFrame(rows, index=["RNN", "LSTM", "Delta (LSTM-RNN)"]).T.to_string(float_format=lambda x: f"{x:,.3f}"))

    # ---------- interpretability ----------
    close_full = feat["Close"]; episodes = I.find_episodes(close_full)
    print("\n=== B&H drawdown zones (>=%d%%) ===" % (C.DD_MIN_DEPTH * 100)); [print(f"  {e['label']:<18} {e['peak'].date()} -> {e['trough'].date()}  {e['depth']*100:.1f}%") for e in episodes]
    gate = I.gate_analysis(wf["gates"], oos, feat)
    print("\n(a) forget-gate vs trend (pooled Pearson):", {k: round(v["pearson"], 3) for k, v in gate["pooled"].items()})
    print("    median |r| cell-state PC1 per fold:", {k: round(v, 3) for k, v in gate["median_abs_pc1"].items()})
    crash = I.crash_table(bt, episodes, pd.Timestamp(oos["Next_Date"].iloc[0]), rnn_eq, exposure)
    print("\n(b) crash windows (OOS part only; flat day = exposure < 50%):\n" + crash.to_string(float_format=lambda x: f"{x:,.2f}"))
    abl = I.ablation_table(oos, wf["ablation"])
    print("\n(c) FEATURE-GROUP IMPORTANCE (same trained models, group neutralised / time-shuffled at test time):\n" + abl.round(3).to_string())
    label, text = I.verdict(gate, crash, abl, lstm_m, rnn_m, h_pm, n_lstm, n_rnn, str(ins_end.date()), exposure,
                            exposure * 100, med_epochs, em_m, skill)
    print("\n" + text)

    # ---------- save ----------
    oos.to_csv(P("oos_predictions_longterm_lstm.csv"))
    bt.assign(close_t=oos["Close"].values).to_csv(P("equity_longterm_lstm.csv"))
    g = wf["gates"]; np.savez_compressed(P("gates_cellstate_longterm_lstm.npz"), dates=oos.index.values.astype("datetime64[ns]"), fold=oos["Fold"].values, **g)
    out = {"experiment": "longterm_daily", "model": "LSTM (v5 position-sizing)", "ticker": C.TICKER,
           "config": {k: getattr(C, k) for k in ("SEED", "WINDOW", "HIDDEN", "NUM_LAYERS", "DROPOUT", "BATCH", "LR", "EPOCHS", "PATIENCE", "GRAD_CLIP",
                                                 "INITIAL_CAPITAL", "COST_PER_SIDE", "INITIAL_TRAIN_YEARS", "RETRAIN_EVERY", "N_SEEDS", "LOSS",
                                                 "RISK_KAPPA", "REBAL_BAND", "AUX_WEIGHT", "USE_EXOG")},
           "features": C.FEATURES, "feature_groups": C.FEATURE_GROUPS,
           "version": "v5: stationary multi-source features, utility-trained exposure head + aux return head, deadband, benchmarks, bootstrap",
           "n_params": {"lstm": n_lstm, "simple_rnn": n_rnn}, "average_exposure": exposure,
           "in_sample_window": [str(ins_start.date()), str(ins_end.date())], "oos_window": [str(oos_start.date()), str(feat.index[-1].date())],
           "metrics": {"strategy": lstm_m, "buy_hold": bh_m, "exposure_matched_bh": em_m, "rules": rules_m, "persistence": raw_m["persistence"],
                       "skill": skill, "median_best_epoch": med_epochs},
           "sharpe_diff_vs_buy_hold_bootstrap": boot, "scorecard": {"label": sc_label, "text": sc_text, **sc_flags},
           "folds": wf["folds"], "crash_windows": crash.astype(object).where(crash.notna(), None).to_dict("records"),
           "gate_analysis": {k: v for k, v in gate.items() if not k.endswith("_series")}, "feature_group_ablation": abl.to_dict("index"),
           "verdict_200d_memory": {"label": label, "text": text}, "smoke_test_synthetic_data": smoke}
    M.save_json(out, P("results_longterm_lstm.json"))
    extras = {nm: pd.Series(v["equity"], index=bt.index) for nm, v in rules.items()}
    plots.plot_equity(bt, close_full, episodes, oos_start, rnn_eq, P("equity_curve_longterm_lstm.png"), P("equity_curve_longterm_lstm.html"), extras=extras)
    plots.plot_gates(oos, feat, gate, episodes, P("forget_gate_longterm_lstm.png"), P("forget_gate_longterm_lstm.html"))
    plots.plot_weights(bt, episodes, P("exposure_longterm_lstm.png"))
    print(f"\nSaved results/figures to ./{C.OUT_DIR}/")
    print("\nLIMITATIONS: (1) single asset, single path, ~17 retrain folds - one realisation, not proof; the bootstrap CI covers sampling noise in this path only. "
          "(2) No model or feature set can guarantee beating Buy & Hold: a long-only strategy capped at 100% exposure normally earns less total return than "
          "Buy & Hold in a bull market, so success is judged on risk-adjusted terms (Sharpe, Calmar, drawdown) against Buy & Hold AND simple rules. "
          "(3) Initial window covers 2008 in-sample; features start after the 252-day warm-up. (4) Macro/cross-asset series (VIX, yields, credit, bonds, gold) "
          "are taken at the close of day t from Yahoo (adjusted, forward-filled); data revisions and vendor errors are possible. (5) Gate/cell-state units are "
          "not comparable across retrained folds; correlations of trending series can be spurious. (6) Costs are 0.05%/side of traded notional only: no slippage, "
          "taxes, spread, financing or FX (USD returns applied to Rs capital); cash earns 0. (7) The 2007-2026 US-equity regime (long bull market, V-shaped "
          "recoveries) is one regime; no other assets or markets tested. (8) Heuristic verdict thresholds. Not investment advice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LSTM long-term daily experiment")
    ap.add_argument("--ticker", default=None, help="asset to trade, default from config (SPY). Use ^GSPC for the raw index")
    ap.add_argument("--smoke", action="store_true", help="offline synthetic-data plumbing test (not a result)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None, help="auto = GPU if available (default)")
    ap.add_argument("--seeds", type=int, default=None, help="ensemble size per fold (default 3)")
    ap.add_argument("--no-exog", action="store_true", help="skip VIX / rates / credit / gold / bond features")
    a = ap.parse_args()
    if a.ticker: C.set_ticker(a.ticker)
    if a.device: C.DEVICE = a.device
    if a.seeds: C.N_SEEDS = a.seeds
    if a.no_exog: C.USE_EXOG = False
    main(smoke=a.smoke)
