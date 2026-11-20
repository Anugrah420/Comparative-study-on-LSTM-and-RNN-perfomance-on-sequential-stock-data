"""
config.py — Fixed constants for the LSTM-on-INTRADAY experiment.

Every value here is IDENTICAL to the Simple RNN intraday experiment.
The ONLY difference between the two experiments is the recurrent cell
(nn.RNN -> nn.LSTM), which lives in models.py. Do not change these values,
or the A/B comparison is no longer like-for-like.
"""
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Asset / data ──────────────────────────────────────────────────────────
TICKER = "AAPL"                 # strictly Apple Inc. No other tickers.
INTERVAL = "5m"
PERIOD = "60d"                  # Yahoo's maximum look-back for 5m bars
BAR_MINUTES = 5
MARKET_TZ = "America/New_York"
SESSION_START = "09:30"         # first regular-session bar (bar START time)
SESSION_LAST_BAR = "15:55"      # last bar starts 15:55 and closes at 16:00
BARS_PER_DAY = 78               # 6.5 h * 12 bars/h

# ── Capital / costs ───────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000       # ₹1,00,000
COST_PER_SIDE = 0.0005          # 0.05 % per side
ROUND_TRIP_COST = 2 * COST_PER_SIDE   # trade threshold for the signal rule

# ── Model / training (IDENTICAL to the RNN experiment) ────────────────────
WINDOW = 48                     # bars per input sequence (4 h of 5m bars)
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH = 128
LR = 1e-3
EPOCHS = 60
PATIENCE = 8                    # early stopping on validation loss
GRAD_CLIP = 1.0
SHUFFLE_TRAIN_BATCHES = True    # mini-batch order inside TRAIN only; the
                                # train/val/test split itself is chronological.

# ── Split ─────────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15                 # test = remaining 0.15

# ── Features (EXACT SAME micro-features as the RNN experiment) ────────────
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA_9", "VWAP", "RSI_14", "ATR_14",
    "Minute_of_Day_sin", "Minute_of_Day_cos",
]
TARGET = "Close"                # next candle's Close (modelled as log return, reported in USD)
VOLUME_FEATURE_IDX = FEATURES.index("Volume")

# ── Metrics ───────────────────────────────────────────────────────────────
ANN_FACTOR = 252 * 78           # bars per year for Sharpe/Sortino
RF_ANNUAL = 0.0
VAR_PCTL = 5                    # historical VaR 95 % -> 5th percentile

# ── Diagnostics ───────────────────────────────────────────────────────────
VOL_STD_WINDOW = 12             # rolling 12-bar return std (volatility regime)
VOL_PCTL = 75                   # high-vol = above TRAIN-set 75th percentile
SPIKE_Z = 2.0                   # volume spike = time-of-day-adjusted z > 2
ABLATION_REPEATS = 5
N_PERMUTATIONS = 2000

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
EXPERIMENT = "intraday_lstm"
RESULTS_JSON = OUT_DIR / "results_intraday_lstm.json"
PREDICTIONS_CSV = OUT_DIR / "test_predictions_intraday_lstm.csv"
GATES_NPZ = OUT_DIR / "gate_activations_intraday_lstm.npz"
WEIGHTS_PT = OUT_DIR / "lstm_intraday_weights.pt"
RAW_CACHE_CSV = OUT_DIR / "raw_aapl_5m.csv"   # snapshot of the downloaded data

# Where the sibling Simple RNN experiment writes its outputs (optional).
RNN_OUT_DIR = ROOT.parent / "RNN on intraday" / "outputs"
RNN_RESULTS_JSON_NAME = "results_intraday_rnn.json"
RNN_PREDICTIONS_CSV_NAME = "test_predictions_intraday_rnn.csv"
