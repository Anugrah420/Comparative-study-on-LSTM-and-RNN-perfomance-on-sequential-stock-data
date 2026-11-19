# LSTM on intraday (AAPL, 5-minute bars)

This is the LSTM half of the Simple RNN vs LSTM A/B test. Everything is the same as the RNN intraday experiment except the recurrent cell.

## Layout
```
LSTM on intraday/
├── config.py        constants (identical to the RNN experiment)
├── data.py          download → clean → features → split → scale → windows → leakage asserts
├── models.py        LSTMRegressor, SimpleRNNRegressor, param counts, gate extraction
├── train.py         seeding, training with early stopping, predict
├── backtest.py      signal rule, intraday backtest, buy & hold
├── metrics.py       evaluation matrix, ₹ formatting, JSON, RNN vs LSTM table
├── diagnostics.py   forget-gate/volume, VWAP, volatility regimes, volume ablation, verdict
├── plots.py         3-day chart, equity curve, gate vs volume, training curve
├── main.py          runs the whole pipeline
└── requirements.txt
```
Output files go to `outputs/`, and figures go to `outputs/figures/`.

## Run
```bash
cd "LSTM on intraday"
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                      # downloads 60 days of data and saves outputs/raw_aapl_5m.csv
python main.py --param-matched      # adds the parameter-matched LSTM robustness row
```

## Pairing with the RNN run (important)
Yahoo only serves 60 days of 5-minute data, so the data changes every day. Both models must see the same data:
1. Put the two folders side by side: `RNN on intraday/` and `LSTM on intraday/`.
2. Run one experiment first. It saves `outputs/raw_aapl_5m.csv`.
3. Run the other one with `--data-csv "../RNN on intraday/outputs/raw_aapl_5m.csv"` (or the reverse).
4. The RNN run should write `outputs/results_intraday_rnn.json`, using the same schema as `metrics.py`, and `outputs/test_predictions_intraday_rnn.csv`, with `target_timestamp` and `pred_next` columns. If these files exist, the LSTM run adds the side-by-side table, the dashed RNN line and the RNN reference values in the verdict. Use `--rnn-dir` if the folder is somewhere else.
