"""Compare RNN and LSTM results across seeds.

Usage:  python compare.py [prefix]      # default prefix: outputs_seed
Reads <prefix><seed>/<cell>/{metrics,history,config}.json for seeds 42, 1, 2, 3.
"""
import json
import sys
from pathlib import Path

import pandas as pd

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "outputs_seed"
SEEDS = (42, 1, 2, 3)
CELLS = ("rnn", "lstm")
COLS = ["best_epoch", "rmse", "rmse_zero", "corr", "directional_acc",
        "sharpe", "bh_sharpe", "total_return", "bh_total_return", "max_drawdown", "n_trades"]

rows = []
for s in SEEDS:
    cfgs = {}
    for c in CELLS:
        d = Path(f"{PREFIX}{s}") / c
        if not (d / "metrics.json").exists():
            print(f"[missing] {d}")
            continue
        m = json.loads((d / "metrics.json").read_text())
        h = json.loads((d / "history.json").read_text())
        cfgs[c] = json.loads((d / "config.json").read_text())
        r = {"seed": s, "cell": c, "best_epoch": h["best_epoch"]}
        r.update(m["regression"]); r.update(m["trading"])
        rows.append(r)
    if len(cfgs) == 2:
        a, b = cfgs["rnn"], cfgs["lstm"]
        if a["fingerprint"] != b["fingerprint"] or a["data_hash"] != b["data_hash"]:
            print(f"[WARNING] seed {s}: fingerprint or data_hash differs -> NOT a fair comparison")

if not rows:
    sys.exit("No results found.")

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== Per run ===")
print(df.set_index(["seed", "cell"])[COLS].round(4))
print("\n=== Mean / std across seeds ===")
print(df.groupby("cell")[COLS].agg(["mean", "std"]).round(4).T)
df.to_csv("model_comparison.csv", index=False)