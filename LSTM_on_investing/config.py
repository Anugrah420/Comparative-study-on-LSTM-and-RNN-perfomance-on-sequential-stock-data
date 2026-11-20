"""Fixed constants for the LONG-TERM DAILY LSTM experiment (identical to the RNN counterpart).
Every other module does `import config as C` and reads C.<NAME> at call time."""
# ---- reproducibility / model / training (DO NOT TUNE DIFFERENTLY FROM THE RNN) ----
SEED = 42
WINDOW = 60
HIDDEN = 32                       # v5: smaller (more inputs, little data)
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH = 64
LR = 3e-4                         # v3: was 1e-3; best epoch was 1 in many folds (overfits noise immediately)
EPOCHS = 80
PATIENCE = 10
GRAD_CLIP = 1.0
# ---- market / capital ----
TICKER = "SPY"                    # S&P 500 ETF (tradable, dividends included via auto_adjust). "^GSPC" = the raw index (price only). CLI: --ticker
TICKERS_NOTE = "SPY tracks the S&P 500; ^GSPC is the non-tradable price index"
INITIAL_CAPITAL = 100000          # Rs 1,00,000
COST_PER_SIDE = 0.0005            # 0.05 % per side
ROUND_TRIP_COST = 2 * COST_PER_SIDE
TRADING_DAYS = 252
RF_ANNUAL = 0.0
# ---- walk-forward ----
INITIAL_TRAIN_YEARS = 3           # ~3*252 rows of post-warm-up data
RETRAIN_EVERY = 252
VAL_FRAC = 0.15
# ---- features ----
FEATURES = ["Ret_1d", "Ret_5d", "Ret_20d", "HL_Range", "Gap", "Intraday", "Vol_Rel",
            "Close_SMA50", "Close_SMA200", "SMA200_Slope", "MACD_N", "MACD_Sig_N",
            "Rolling_Std_20", "Month_sin", "Month_cos"]          # all stationary (v2)
TREND_FEATURES = ["Close_SMA200", "SMA200_Slope"]               # the 200-day information (ablated in test (c))
# ---- analysis ----
RUN_PARAM_MATCHED = False          # robustness variant, clearly labelled in all outputs
DD_MIN_DEPTH = 0.15               # B&H drawdown episodes deeper than 15 % are "crash zones" (index drawdowns are shallower than single stocks')
DD_TOP_K = 6
MIN_CRASH_DAYS = 15               # min OOS days in a crash window to be scored
# ---- files ----
OUT_DIR = "outputs_longterm_lstm_spy"


def safe_name(t):
    return "".join(ch for ch in t.lower() if ch.isalnum())


def set_ticker(t):
    """Switch asset and its output folder (e.g. 'SPY' -> outputs_longterm_lstm_spy, '^GSPC' -> ..._gspc)."""
    global TICKER, OUT_DIR
    TICKER = t; OUT_DIR = f"outputs_longterm_lstm_{safe_name(t)}"
RNN_RESULTS = "results_longterm_rnn.json"
RNN_EQUITY = "equity_longterm_rnn.csv"   # needs columns: date, equity (optional: position)

# ---- v2 modifications (fix for the degenerate v1 run) ----
N_SEEDS = 3                       # ensemble: average predictions of N_SEEDS independently-seeded LSTMs per fold
LOSS = "huber"                    # Huber(delta=1) on standardised returns (fat tails); "mse" to revert
MIN_EXPOSURE = 0.10               # verdict flags the run as degenerate if time-in-market < 10 %

# ---- v3 modifications ----
DEVICE = "auto"                   # "auto" (CUDA if available else CPU) | "cuda" | "cpu"   (CLI: --device)
WEIGHT_DECAY = 1e-4               # AdamW decoupled weight decay (regularisation)
FORGET_BIAS_INIT = 1.0            # forget-gate bias init (standard LSTM trick: gates start 'remembering', sigmoid(1)=0.73)
MIN_LAST_FOLD = 63                # a final OOS block shorter than this is merged into the previous fold

# ---- v5: feature-rich, position-sizing LSTM (defaults to Buy & Hold, deviates only when features justify it) ----
USE_EXOG = True                   # download VIX, 10y yield, HYG/LQD, TLT, GLD (CLI: --no-exog). Missing series are skipped automatically.
AUX_WEIGHT = 1.0                  # weight of the auxiliary next-day-return head (keeps DA / RMSE / MAPE reportable)
RISK_KAPPA = 1.0                  # risk aversion multiplier: 1.0 -> unconditional optimal exposure = 100 % (= Buy & Hold); >1 = more cautious
REBAL_BAND = 0.10                 # only trade when target exposure differs from current by >= 10 pp (cuts turnover/cost)
W_INIT_LOGIT = 2.2                # head bias init: sigmoid(2.2) = 0.90 exposure at start
VOL_TARGET = 0.10                 # simple benchmark rule: 10 % annualised volatility targeting
FEATURE_GROUPS = {}               # filled by data.build_features (used for group ablation / importance)
