"""Experiment: SIMPLE RNN on LONG-TERM DAILY AAPL data (one arm of the RNN-vs-LSTM A/B).

Run:
    python main.py                      # live AAPL via yfinance (needs internet)
    python main.py --csv aapl.csv       # same pipeline on a saved AAPL OHLCV CSV
    python main.py --synthetic          # offline smoke test ONLY (fake prices)
    python main.py --model lstm         # the LSTM counterpart, identical settings

Every hyperparameter below is shared with the LSTM experiment; only the
recurrent cell changes, so any difference in results is attributable to it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

# ============================ GLOBAL CONSTANTS ============================ #
SEED = 42
WINDOW = 60                 # look-back, trading days
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH = 64
LR = 1e-3
EPOCHS = 80
PATIENCE = 10               # early stopping on validation loss
GRAD_CLIP = 1.0
INITIAL_CAPITAL = 100_000   # ₹1,00,000
COST_PER_SIDE = 0.0005      # 0.05% per side; round trip = 0.10%
RF_ANNUAL = 0.0             # risk-free rate for Sharpe/Sortino (same in all experiments)
YEARS = 20
INITIAL_TRAIN_DAYS = 3 * 252   # ~3 years after indicator warm-up
RETRAIN_EVERY = 252            # retrain yearly, predict the next 252 days
VAL_FRACTION = 0.15            # last 15% of each training window -> early stopping
SCALING = "window_relative"    # see walkforward.make_windows ("global" = spec-literal)
REGIME = "longterm_daily"
# ========================================================================== #

import data as data_mod                                    # noqa: E402
from backtest import run_backtest                          # noqa: E402
from metrics import (directional_accuracy, drawdown_episodes, fit_metrics,  # noqa: E402
                     format_inr, max_drawdown)
from models import build_model, count_parameters           # noqa: E402
from plots import equity_drawdown_figure, fold_loss_figure, gradient_decay_figure  # noqa: E402
from train import (get_device, gradient_norm_profile, predict, set_determinism,  # noqa: E402
                   summarise_gradient_profile, train_model)
from walkforward import run_walk_forward                   # noqa: E402

MODEL_NAMES = {"rnn": "SimpleRNN", "lstm": "LSTM"}

LIMITATIONS = (
    "Limitations: (1) single asset — AAPL only, so nothing here generalises across "
    "stocks or sectors; (2) survivorship/outlier bias — AAPL is one of the best-"
    "performing large caps of the last 20 years, so Buy & Hold is an unusually hard "
    "benchmark and any long-only result is flattered by the asset choice; (3) regime "
    "sparsity — only a handful of macro crashes exist in 20 years, and 2008 falls in "
    "the in-sample training region, so crash conclusions rest on very few events; "
    "(4) 1-day-ahead price-level targets sit very close to persistence (tomorrow ≈ "
    "today), so low RMSE/MAPE mostly reflects that, not skill — compare against the "
    "persistence row; (5) the backtest ignores taxes, FX (USD/INR held constant), "
    "slippage beyond the flat per-side cost, and market impact."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="AAPL OHLCV CSV (Date,Open,High,Low,Close,Volume)")
    src.add_argument("--synthetic", action="store_true",
                     help="offline smoke test on synthetic prices (not AAPL)")
    p.add_argument("--model", choices=["rnn", "lstm"], default="rnn")
    p.add_argument("--outdir", default=None)
    p.add_argument("--linear", action="store_true", help="linear y-axis on the equity plot")
    p.add_argument("--max-folds", type=int, default=None, help="debug: run only N folds")
    return p.parse_args()


def to_json_safe(obj):
    """Recursively convert numpy/pandas values; NaN -> null, inf -> 'inf'."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return obj


def label_episode(ep: dict) -> str:
    years = range(ep["peak"].year, ep["trough"].year + 1)
    if 2008 in years:
        return "2008 GFC"
    if 2020 in years:
        return "2020 COVID crash"
    if 2022 in years:
        return "2022 drawdown"
    return f"{ep['peak'].year}" + (f"–{ep['trough'].year}" if ep["trough"].year != ep["peak"].year
                                   else "") + " drawdown"


def select_crash_windows(price: pd.Series, top_n: int = 5) -> list[dict]:
    """Deepest B&H drawdowns, plus the deepest 2020 and 2022 episodes if missing."""
    eps = drawdown_episodes(price)                 # deepest first
    special_taken = set()
    for ep in eps:
        ep["label"] = label_episode(ep)
        # Only the deepest episode keeps a named-crash label; shallower ones in
        # the same year get a generic label so annotations are unambiguous.
        if ep["label"] in ("2008 GFC", "2020 COVID crash", "2022 drawdown"):
            if ep["label"] in special_taken:
                ep["label"] = f"{ep['peak']:%b %Y} drawdown"
            special_taken.add(ep["label"])
    chosen = eps[:top_n]
    for must in ("2008 GFC", "2020 COVID crash", "2022 drawdown"):
        if not any(e["label"] == must for e in chosen):
            extra = next((e for e in eps if e["label"] == must and e["depth_pct"] < -10), None)
            if extra:
                chosen.append(extra)
    return sorted(chosen, key=lambda e: e["peak"])


def crash_analysis(windows, bt, oos_start, oos_end) -> list[dict]:
    """(b) strategy vs B&H return and MDD, (c) fraction flat, per crash window."""
    pos, sr, br = bt["position"], bt["strategy_returns"], bt["bh_returns"]
    overall_flat = float((pos == 0).mean() * 100)
    rows = []
    for ep in windows:
        lo, hi = ep["peak"], ep["trough"]
        if hi <= oos_start:
            rows.append({"window": ep["label"], "peak": lo, "trough": hi,
                         "bh_full_history_mdd_pct": ep["depth_pct"],
                         "status": "in-sample (not evaluated)"})
            continue
        mask = (sr.index > max(lo, oos_start)) & (sr.index <= min(hi, oos_end))
        if mask.sum() < 2:
            continue
        s, b = sr[mask], br[mask]
        s_eq, b_eq = np.concatenate([[1], (1 + s).cumprod()]), np.concatenate([[1], (1 + b).cumprod()])
        rows.append({
            "window": ep["label"], "peak": lo, "trough": hi, "days": int(mask.sum()),
            "status": "out-of-sample" if lo >= oos_start else "partly out-of-sample",
            "bh_full_history_mdd_pct": ep["depth_pct"],
            "strategy_return_pct": (s_eq[-1] - 1) * 100, "bh_return_pct": (b_eq[-1] - 1) * 100,
            "strategy_mdd_pct": max_drawdown(s_eq) * 100, "bh_mdd_pct": max_drawdown(b_eq) * 100,
            "flat_fraction_pct": float((pos[mask] == 0).mean() * 100),
            "overall_flat_fraction_pct": overall_flat,
        })
    return rows


def verdict(grad_summary: dict, crashes: list[dict], model_name: str) -> tuple[str, str]:
    """Rule-based verdict on 'the RNN suffers vanishing gradients and fails to adapt'."""
    ratio = grad_summary["mean_top_layer_ratio"]
    memory = grad_summary["mean_effective_memory_steps"]
    vanishing = "strong" if ratio < 1e-2 else ("moderate" if ratio < 1e-1 else "none")
    # Only windows that lie entirely out-of-sample count as evidence.
    oos = [c for c in crashes if c["status"] == "out-of-sample"]
    outcomes = []
    for c in oos:
        protection = 1 - abs(c["strategy_mdd_pct"]) / max(abs(c["bh_mdd_pct"]), 1e-9)
        derisked = c["flat_fraction_pct"] >= c["overall_flat_fraction_pct"] + 10
        if protection < 0.2 and not derisked:
            outcomes.append("failed")
        elif protection > 0.5 and derisked:
            outcomes.append("adapted")
        else:
            outcomes.append("mixed")
    failed, adapted = outcomes.count("failed"), outcomes.count("adapted")

    if not oos:
        label = "INCONCLUSIVE"
    elif vanishing == "strong" and failed > len(oos) / 2:
        label = "SUPPORTED"
    elif vanishing == "none" and failed == 0:
        label = "NOT SUPPORTED"
    elif vanishing != "strong" and adapted > len(oos) / 2:
        label = "NOT SUPPORTED"
    else:
        label = "INCONCLUSIVE"

    crash_txt = "; ".join(
        f"{c['window']}: strategy {c['strategy_return_pct']:+.1f}% vs B&H {c['bh_return_pct']:+.1f}%, "
        f"MDD {c['strategy_mdd_pct']:.1f}% vs {c['bh_mdd_pct']:.1f}%, flat {c['flat_fraction_pct']:.0f}% "
        f"of days (overall {c['overall_flat_fraction_pct']:.0f}%) → {o}"
        for c, o in zip(oos, outcomes)) or "no crash window falls in the out-of-sample period"
    text = (
        f"Verdict: {label}. Gradient diagnostic — across folds the top-layer gradient at the "
        f"oldest step is {ratio:.2e}× the most-recent step ({vanishing} decay; effective memory "
        f"≈{memory:.0f} of {WINDOW} steps at a 1% threshold); the input-sensitivity ratio "
        f"‖∂L/∂x₁‖/‖∂L/∂x₆₀‖ is {grad_summary['mean_input_grad_ratio']:.2e}, which is the "
        f"number to compare against the LSTM run. Crash behaviour (out-of-sample "
        f"only): {crash_txt}. 'Supported' requires strong decay AND failure in most "
        f"out-of-sample crashes; a model that is flat most of the time is not credited with "
        f"crash protection unless it is flat more often during crashes than overall. Caveat: "
        f"the 2008 GFC lies inside the in-sample initial-training region, so it cannot count as "
        f"evidence either way, and decay alone does not prove crash failure — compare with the "
        f"LSTM run (python main.py --model lstm) before attributing differences to the cell."
    )
    return label, text


def main():
    args = parse_args()
    cell = args.model
    model_name = MODEL_NAMES[cell]
    tag = f"longterm_{cell}"
    outdir = args.outdir or os.path.join("outputs", tag)
    os.makedirs(outdir, exist_ok=True)
    set_determinism(SEED)
    device = get_device()
    t0 = time.time()

    # ------------------------------ data ------------------------------ #
    if args.synthetic:
        raw, source, attempts = data_mod.synthetic_aapl_like(YEARS, seed=SEED), \
            "SYNTHETIC (offline smoke test — not AAPL)", []
    elif args.csv:
        raw, source, attempts = data_mod.load_csv(args.csv), f"CSV {args.csv}", []
    else:
        raw, source, attempts = data_mod.download_aapl(YEARS)
    cov = data_mod.coverage_report(raw)
    print(f"[data] source: {source}")
    for line in attempts:
        print(f"       attempt — {line}")
    print(f"[data] {cov['rows']} rows, {cov['start']} → {cov['end']} ({cov['span_years']} years); "
          f"covers 2008: {cov['covers_2008']}, covers 2020: {cov['covers_2020']}")
    if not (cov["covers_2008"] and cov["covers_2020"]):
        print("[data] WARNING: 2008 and/or 2020 are not covered; crash analysis will be partial.")

    feat, finfo = data_mod.build_features(raw)
    print(f"[features] {len(data_mod.FEATURE_COLS)} inputs; warm-up rows dropped: "
          f"{finfo['warmup_rows_dropped']} (SMA_200), final row dropped: 1 (no next close); "
          f"usable rows: {finfo['usable_rows']} from {finfo['first_usable_date']}")

    hp = dict(WINDOW=WINDOW, SCALING=SCALING, INITIAL_TRAIN_DAYS=INITIAL_TRAIN_DAYS,
              RETRAIN_EVERY=RETRAIN_EVERY, VAL_FRACTION=VAL_FRACTION)
    n_feat = len(data_mod.FEATURE_COLS)
    probe = build_model(n_feat, cell, HIDDEN, NUM_LAYERS, DROPOUT)
    print(f"[model] {model_name}: nn.{'RNN (tanh)' if cell == 'rnn' else 'LSTM'}, "
          f"{NUM_LAYERS} layers × {HIDDEN} hidden, dropout {DROPOUT}, linear head on last step; "
          f"{count_parameters(probe):,} trainable parameters; device {device}")

    # ------------------------- walk-forward --------------------------- #
    grad_profiles: dict[int, np.ndarray] = {}
    input_profiles: dict[int, np.ndarray] = {}

    def fit_fold(k, X_fit, y_fit, X_val, y_val, X_test):
        set_determinism(SEED)                       # identical init for every fold
        model = build_model(n_feat, cell, HIDDEN, NUM_LAYERS, DROPOUT)
        hist = train_model(model, X_fit, y_fit, X_val, y_val, epochs=EPOCHS, batch=BATCH,
                           lr=LR, patience=PATIENCE, clip=GRAD_CLIP, seed=SEED,
                           device=device, log_prefix=f"  [fold {k}]")
        grad_profiles[k], input_profiles[k] = gradient_norm_profile(model, X_fit, y_fit,
                                                                    device, seed=SEED)
        return predict(model, X_test, device), hist

    oos, fold_logs = run_walk_forward(raw, feat, hp, fit_fold, max_folds=args.max_folds)
    print("\n[leakage] assert_no_lookahead passed on every fold "
          "(indicators recomputed on truncated data; scalers fitted on training rows only)")
    oos_start, oos_end = oos["decision_date"].iloc[0], oos["next_date"].iloc[-1]

    # --------------------------- backtest ----------------------------- #
    bt = run_backtest(oos, INITIAL_CAPITAL, COST_PER_SIDE, RF_ANNUAL)
    sm, bm = bt["strategy_metrics"], bt["bh_metrics"]
    sm.update(fit_metrics(oos["actual_next_close"], oos["pred_next_close"]))
    persistence = fit_metrics(oos["actual_next_close"], oos["close_t"])
    persistence["directional_accuracy_pct"] = directional_accuracy(
        oos["close_t"], oos["close_t"], oos["actual_next_close"])  # undefined -> NaN

    # ------------------------ evaluation matrix ----------------------- #
    def pct(v):
        return "—" if v is None or not np.isfinite(v) else f"{v:,.2f}%"

    def num(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return "∞" if np.isinf(v) else f"{v:,.2f}"

    rows = [
        ("Capital & Profit", "Total ROI (%)", pct(sm["total_roi_pct"]), pct(bm["total_roi_pct"]), "—"),
        ("Capital & Profit", "Net Profit (₹)", format_inr(sm["net_profit_inr"]),
         format_inr(bm["net_profit_inr"]), "—"),
        ("Execution", "Directional Accuracy (%)", pct(sm["directional_accuracy_pct"]), "—",
         "— (predicts no change)"),
        ("Execution", "Profit Factor", num(sm["profit_factor"]), num(bm["profit_factor"]), "—"),
        ("Execution", "Total Trades Executed", f"{sm['total_trades']}", "1", "—"),
        ("Risk", "Maximum Drawdown (%)", pct(sm["max_drawdown_pct"]), pct(bm["max_drawdown_pct"]), "—"),
        ("Risk", "VaR 95% daily (%)", pct(sm["var95_pct"]), pct(bm["var95_pct"]), "—"),
        ("Risk", "VaR 95% daily (₹, on initial capital)",
         format_inr(sm["var95_inr_on_initial_capital"]),
         format_inr(bm["var95_inr_on_initial_capital"]), "—"),
        ("Risk-Adjusted", "Sharpe Ratio", num(sm["sharpe"]), num(bm["sharpe"]), "—"),
        ("Risk-Adjusted", "Sortino Ratio", num(sm["sortino"]), num(bm["sortino"]), "—"),
        ("ML Predictive Fit", "RMSE (USD)", num(sm["rmse_usd"]), "—", num(persistence["rmse_usd"])),
        ("ML Predictive Fit", "MAPE (%)", pct(sm["mape_pct"]), "—", pct(persistence["mape_pct"])),
    ]
    matrix = pd.DataFrame(rows, columns=["Category", "Metric", f"{model_name} Strategy",
                                         "Buy & Hold", "Persistence baseline"])
    pd.set_option("display.width", 200)
    print(f"\n=== EVALUATION MATRIX — {model_name}, long-term daily, out-of-sample "
          f"{oos_start.date()} → {oos_end.date()} ===")
    print(matrix.to_string(index=False))
    print(f"(Exposure {sm['exposure_pct']:.1f}% of days long; signal threshold "
          f"{bt['signal_threshold'] * 100:.2f}% = round-trip cost. FX-neutral: ₹ equity = "
          f"{format_inr(INITIAL_CAPITAL, 0)} × cumulative growth of a fractional USD position.)")

    # -------------------- crash + gradient diagnostics ---------------- #
    price = feat["Close"]
    windows = select_crash_windows(price)
    crashes = crash_analysis(windows, bt, oos_start, oos_end)
    print("\n=== CRASH WINDOWS (peak → trough of Buy & Hold) ===")
    crash_df = pd.DataFrame(crashes).drop(columns=["overall_flat_fraction_pct"], errors="ignore")
    for col in ("peak", "trough"):
        crash_df[col] = pd.to_datetime(crash_df[col]).dt.date
    print(crash_df.round(2).to_string(index=False))
    print(f"(Overall fraction of out-of-sample days flat: "
          f"{float((bt['position'] == 0).mean() * 100):.1f}%. Only fully out-of-sample "
          f"windows count toward the verdict.)")

    per_fold = [summarise_gradient_profile(grad_profiles[k], input_profiles[k])
                for k in grad_profiles]
    grad_summary = {
        "mean_top_layer_ratio": float(np.exp(np.mean(np.log(
            [max(s["grad_ratio_t1_over_tW"], 1e-30) for s in per_fold])))),  # geometric mean
        "mean_input_grad_ratio": float(np.exp(np.mean(np.log(
            [max(s["input_grad_ratio_t1_over_tW"], 1e-30) for s in per_fold])))),
        "mean_effective_memory_steps": float(np.mean([s["effective_memory_steps_1pct"] for s in per_fold])),
        "per_fold": per_fold,
        "mean_profile_top_layer": np.stack(list(grad_profiles.values()))[:, -1].mean(0).tolist(),
        "mean_profile_input": np.stack(list(input_profiles.values())).mean(0).tolist(),
    }
    label, verdict_text = verdict(grad_summary, crashes, model_name)
    print("\n=== VANISHING-GRADIENT / CRASH-ADAPTATION VERDICT ===")
    print(verdict_text)
    print("\n" + LIMITATIONS)

    # ------------------------------ save ------------------------------ #
    oos_out = oos.copy()
    oos_out["position_next_day"] = bt["position"].to_numpy()
    oos_out["strategy_return"] = bt["strategy_returns"].to_numpy()
    oos_out["strategy_equity_inr"] = bt["strategy_equity"].to_numpy()
    oos_out["bh_equity_inr"] = bt["bh_equity"].to_numpy()
    oos_out.to_csv(os.path.join(outdir, f"oos_predictions_{tag}.csv"), index=False)

    results = {
        "regime": REGIME, "model": model_name,
        "metrics": sm,
        "benchmarks": {"buy_and_hold": bm, "persistence": persistence},
        "evaluation_period": {"oos_start": oos_start, "oos_end": oos_end,
                              "in_sample_region": [feat.index[0], oos_start]},
        "crash_analysis": crashes,
        "vanishing_gradient": grad_summary,
        "verdict": {"label": label, "text": verdict_text},
        "folds": [{k: v for k, v in log.items() if k not in ("train_loss", "val_loss")}
                  for log in fold_logs],
        "data": {"source": source, "coverage": cov, "features": finfo,
                 "feature_cols": data_mod.FEATURE_COLS},
        "hyperparameters": {"SEED": SEED, "WINDOW": WINDOW, "HIDDEN": HIDDEN,
                            "NUM_LAYERS": NUM_LAYERS, "DROPOUT": DROPOUT, "BATCH": BATCH,
                            "LR": LR, "EPOCHS": EPOCHS, "PATIENCE": PATIENCE,
                            "GRAD_CLIP": GRAD_CLIP, "OPTIMIZER": "Adam", "LOSS": "MSE",
                            "INITIAL_TRAIN_DAYS": INITIAL_TRAIN_DAYS,
                            "RETRAIN_EVERY": RETRAIN_EVERY, "VAL_FRACTION": VAL_FRACTION,
                            "SCALING": SCALING},
        "assumptions": {"initial_capital_inr": INITIAL_CAPITAL, "cost_per_side": COST_PER_SIDE,
                        "rf_annual": RF_ANNUAL,
                        "fx": "FX-neutral: ₹ equity = INITIAL_CAPITAL × cumulative growth of a "
                              "fractional-share USD position; USD/INR changes ignored.",
                        "position_rule": "long next day if pred > close × (1 + 2×cost), else flat"},
        "limitations": LIMITATIONS,
    }
    with open(os.path.join(outdir, f"results_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(to_json_safe(results), fh, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, f"evaluation_matrix_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {model_name} — long-term daily AAPL\n\nData: {source}. Out-of-sample "
                 f"{oos_start.date()} → {oos_end.date()}.\n\n")
        fh.write(matrix.to_markdown(index=False) if _has_tabulate() else matrix.to_string(index=False))
        fh.write(f"\n\n{verdict_text}\n\n{LIMITATIONS}\n")

    equity_drawdown_figure(price, bt, windows, {"2008 GFC", "2020 COVID crash", "2022 drawdown"},
                           INITIAL_CAPITAL, oos_start, model_name,
                           os.path.join(outdir, f"equity_drawdown_{tag}.png"),
                           os.path.join(outdir, f"equity_drawdown_{tag}.html"),
                           log_scale=not args.linear)
    gradient_decay_figure(grad_profiles, input_profiles, model_name,
                          os.path.join(outdir, f"gradient_decay_{tag}.png"),
                          os.path.join(outdir, f"gradient_decay_{tag}.html"))
    fold_loss_figure(fold_logs, model_name, os.path.join(outdir, f"fold_losses_{tag}.png"),
                     os.path.join(outdir, f"fold_losses_{tag}.html"))
    print(f"\n[done] {len(fold_logs)} folds in {time.time() - t0:.0f}s; outputs in {outdir}/")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
