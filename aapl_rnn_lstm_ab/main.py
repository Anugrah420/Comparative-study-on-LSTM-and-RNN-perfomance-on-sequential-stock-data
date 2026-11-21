"""Run the full RNN experiment: data -> train -> predict -> backtest -> metrics -> plots.

Usage:
    python main.py                       # defaults from config.py
    python main.py --epochs 5 --seed 1   # override any Config field
"""
from __future__ import annotations

import argparse
import dataclasses as dc

import pandas as pd

from backtest import run_backtest
from config import Config
from data import FEATURE_COLS, data_hash, load_prices, make_loaders, make_splits
from metrics import regression_metrics, trading_metrics
from models import build_model
from plots import plot_equity, plot_losses, plot_predictions
from train import predict, train
from utils import ensure_dir, get_device, save_json, set_seed


def run_experiment(cfg: Config) -> dict:
    """Run one RNN experiment and save everything to cfg.run_dir. Returns the results dict."""
    run_dir = ensure_dir(cfg.run_dir)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"device: {device} | run dir: {run_dir}")

    df = load_prices(cfg)
    train_ds, val_ds, test_ds, scaler = make_splits(df, cfg)
    print(f"samples -> train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg)

    model = build_model(len(FEATURE_COLS), cfg)
    history = train(model, train_loader, val_loader, cfg)

    preds = scaler.inverse_target(predict(model, test_loader, device))   # raw log returns
    bt = run_backtest(preds, test_ds.y_raw, test_ds.dates, cfg)
    metrics = {
        "regression": regression_metrics(test_ds.y_raw, preds),
        "trading": trading_metrics(bt),
    }

    # ---- save artifacts ----
    save_json({"config": dc.asdict(cfg), "fingerprint": cfg.fingerprint(),
               "data_hash": data_hash(df)}, run_dir / "config.json")
    save_json(history, run_dir / "history.json")
    save_json(metrics, run_dir / "metrics.json")
    pd.DataFrame({"date": test_ds.dates, "y_true": test_ds.y_raw, "y_pred": preds}
                 ).to_csv(run_dir / "predictions.csv", index=False)
    bt.to_csv(run_dir / "backtest.csv")

    plot_losses(history, run_dir)
    plot_predictions(test_ds.y_raw, preds, test_ds.dates, run_dir)
    plot_equity(bt, run_dir)

    return {"history": history, "preds": preds, "backtest": bt, "metrics": metrics}


def _parse_args() -> Config:
    """Build a Config from defaults plus optional --field overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    defaults = Config()
    for f in dc.fields(Config):
        parser.add_argument(f"--{f.name}", type=type(getattr(defaults, f.name)), default=None)
    args = vars(parser.parse_args())
    return dc.replace(defaults, **{k: v for k, v in args.items() if v is not None})


def main() -> None:
    """CLI entry point."""
    cfg = _parse_args()
    results = run_experiment(cfg)
    m = results["metrics"]
    print("\n=== Test metrics (RNN) ===")
    for group, vals in m.items():
        print(f"[{group}]")
        for k, v in vals.items():
            print(f"  {k:>16}: {v:.4f}" if isinstance(v, float) else f"  {k:>16}: {v}")
    print(f"\nSaved to {cfg.run_dir}/")


if __name__ == "__main__":
    main()