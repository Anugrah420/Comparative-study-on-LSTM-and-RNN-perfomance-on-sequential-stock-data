# LSTM position-sizing on the S&P 500 (SPY) - v5

The LSTM does NOT forecast price and then trade a threshold. It outputs a long-only EXPOSURE w in [0,1] each day
(plus an auxiliary next-day-return head so directional accuracy / RMSE / MAPE stay reportable).
It starts at ~90% exposure and is trained with a mean-variance utility whose unconditional optimum is 100% (= Buy & Hold),
so it only moves away from Buy & Hold when its features say risk-adjusted expected return changed.

## Setup (once)
    python -m venv .venv
    .venv\Scripts\activate            (Linux/Mac: source .venv/bin/activate)
    pip install -r requirements.txt
GPU (optional, NVIDIA): install a CUDA build of PyTorch from pytorch.org; check: python -c "import torch; print(torch.cuda.is_available())"

## Run
    python main.py                    full run: SPY, 36 features, 3-seed ensemble, GPU if available (needs internet first time)
    python main.py --seeds 1          ~3x faster
    python main.py --device cuda      force GPU (clear error if unavailable); --device cpu forces CPU
    python main.py --no-exog          price/volume features only (skip VIX/rates/credit/bonds/gold)
    python main.py --ticker "^GSPC"   raw index instead of SPY
    python main.py --smoke            OFFLINE plumbing test on synthetic data - NOT a result

## Features (36 with macro data; all stationary, all known at the close of day t)
price momentum/trend: Ret_1d/5d/21d/63d/126d, Mom_12_1, DD_252, Close_SMA50, Close_SMA200, SMA50_SMA200, SMA200_Slope
volatility/regime:    Std_20, Std_60, Vol_Ratio, Down_Semi_20, Park_20, HL_Range
candle/volume/osc:    Gap, Intraday, Vol_Rel, RSI_14, BB_Z, MACD_N, MACD_Sig_N, Month_sin, Month_cos
VIX:                  VIX_Log, VIX_Chg5, VIX_Rel, VRP (implied minus realised vol)
macro/cross-asset:    TNX_Chg21/63 (10y yield), Credit_Mom21, Credit_Rel (HYG/LQD), TLT_Ret21, GLD_Ret21
A series that cannot be downloaded is skipped automatically and its features are dropped.

## How success is judged (printed after every run)
- Matrix vs Buy & Hold, constant-exposure Buy & Hold, an SMA200 trend filter and 10% vol-targeting - all with the same 0.05%/side cost.
- SCORECARD: Sharpe difference vs Buy & Hold with a 95% block-bootstrap CI, MDD and Calmar. A long-only strategy capped at 100% exposure
  usually earns LESS total return than Buy & Hold in a bull market; "beating" it means better risk-adjusted return.
- FEATURE-GROUP IMPORTANCE table (which feature groups the trained models actually use).
- The earlier 200-day-memory / crash-protection verdict, skill diagnostics and crash-window table.
Outputs go to outputs_longterm_lstm_spy/ (results JSON, predictions, equity, gates npz, equity/forget-gate/exposure figures).
