# RNN Stock Return Experiment

Predicts the **next-day log return** of a stock with a vanilla RNN (PyTorch), then
backtests a simple long/short strategy on the predictions.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py                         # defaults from config.py
python main.py --epochs 5 --seed 1     # override any Config field
python main.py --ticker MSFT --seq_len 60
```

Prices are downloaded with `yfinance` on the first run and cached in `outputs/data/`.

## Outputs (`outputs/rnn/`)

| File | Contents |
|---|---|
| `config.json` | Full config, settings fingerprint, data hash |
| `history.json` | Train/val loss per epoch, best epoch |
| `metrics.json` | Regression and trading metrics on the test set |
| `predictions.csv` | Test dates, actual and predicted returns |
| `backtest.csv` | Signals, positions, returns, equity curves |
| `best_model.pt` | Best weights (lowest validation loss) |
| `losses.png`, `predictions.png`, `equity.png` | Plots |

## Files

| File | Responsibility |
|---|---|
| `config.py` | Immutable `Config` dataclass (all hyperparameters) |
| `utils.py` | Seeding, device selection, directories, JSON helpers |
| `data.py` | Download, features, chronological split, scaling, DataLoaders |
| `models.py` | Vanilla RNN regression model |
| `train.py` | Training loop with early stopping, `predict` |
| `backtest.py` | Long/short strategy with transaction costs |
| `metrics.py` | RMSE, MAE, directional accuracy, Sharpe, drawdown, etc. |
| `plots.py` | Loss, prediction and equity plots |
| `main.py` | Runs the whole pipeline |

Dependencies flow one way: `config`/`utils` → `data`/`models` → `train` → `backtest`/`metrics` → `plots` → `main`.

## Method notes

- **Features:** log return, high-low range, open-close return, volume change, 5-day mean return, 10-day return volatility.
- **Target:** next-day log return, standardised using training statistics.
- **Split:** chronological 70/15/15 (train/val/test), never shuffled across splits. The scaler is fit on the training rows only.
- **Backtest:** at each close go long if the predicted return is above 0, otherwise short, and hold for one day. A cost of `cost_bps` is charged per unit of position change. Buy-and-hold is reported as a benchmark.
- **Baseline:** `rmse_zero` is the RMSE of always predicting 0. A model that does not beat it has learned little.

## Comparing with another model (e.g. an LSTM)

If someone runs a different model on the same data, these must match exactly:
the ticker, start/end dates, split fractions, `seq_len`, seed, training settings
and the price data. Compare `fingerprint` and `data_hash` in each run's `config.json`
(the fingerprint covers all of these settings, so it only matches if they are all equal),
then compare `metrics.json`.

## Caveats

Daily stock returns are extremely noisy, so expect metrics close to the naive baseline.
Backtest results ignore slippage, taxes and shorting costs, and are for learning purposes only.