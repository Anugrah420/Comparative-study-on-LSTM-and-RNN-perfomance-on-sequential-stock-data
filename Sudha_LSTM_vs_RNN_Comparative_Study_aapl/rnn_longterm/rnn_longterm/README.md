# Simple RNN vs LSTM: long-term daily AAPL (Simple RNN arm)

```bash
pip install -r requirements.txt
python main.py                 # Simple RNN on 20y of live AAPL data (yfinance)
python main.py --model lstm    # LSTM counterpart, identical settings
python main.py --csv aapl.csv  # same, from a saved OHLCV CSV
python main.py --synthetic     # offline smoke test only (fake prices, not AAPL)
```

Outputs go to `outputs/longterm_<model>/`: `results_longterm_rnn.json`,
`oos_predictions_longterm_rnn.csv`, `evaluation_matrix_longterm_rnn.md`, and PNG + HTML
figures (equity/drawdown, gradient decay, per-fold losses). The full run takes ~4 min on a CPU.

| File | Role |
|---|---|
| `data.py` | yfinance download with the `"20y"` → start/end → `max` fallbacks, causal indicators |
| `models.py` | `RecurrentRegressor(cell="rnn"/"lstm")` + manual unroll for the gradient diagnostic |
| `train.py` | determinism, Adam/MSE/clipping/early stopping, gradient-norm profiles |
| `walkforward.py` | expanding window (3y initial, retrain every 252 days), per-fold scalers, leakage assertion |
| `backtest.py` / `metrics.py` | generic long/flat backtest and metrics, reused unchanged by every experiment |
| `plots.py` | equity + drawdown with crash shading (log/linear), gradient decay, fold losses |
| `main.py` | fixed constants, orchestration, evaluation matrix, crash analysis, verdict |

Design decision: `SCALING = "window_relative"` (default) expresses prices relative to the
window's last close, and the model predicts log(Close[t+1]/Close[t]). That value is then
converted back to a USD next-day close for RMSE/MAPE. Without this, a scaler fitted on
2007–2010 prices sees 2024 AAPL at roughly 30× its training range, and a tanh RNN cannot
output those values. A crash failure would then come from scaling, not vanishing
gradients. Set `SCALING = "global"` for the spec-literal raw-USD version.
