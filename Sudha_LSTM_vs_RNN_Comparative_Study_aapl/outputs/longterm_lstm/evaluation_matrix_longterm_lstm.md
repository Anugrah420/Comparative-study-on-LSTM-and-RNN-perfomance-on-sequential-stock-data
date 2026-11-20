# LSTM — long-term daily AAPL

Data: yfinance period="20y". Out-of-sample 2010-09-08 → 2026-11-18.

| Category          | Metric                                | LSTM Strategy   | Buy & Hold    | Persistence baseline   |
|:------------------|:--------------------------------------|:----------------|:--------------|:-----------------------|
| Capital & Profit  | Total ROI (%)                         | 1,145.48%       | 4,394.94%     | —                      |
| Capital & Profit  | Net Profit (₹)                        | ₹11,45,482.01   | ₹43,94,940.56 | —                      |
| Execution         | Directional Accuracy (%)              | 52.35%          | —             | — (predicts no change) |
| Execution         | Profit Factor                         | 2.09            | ∞             | —                      |
| Execution         | Total Trades Executed                 | 133             | 1             | —                      |
| Risk              | Maximum Drawdown (%)                  | -39.57%         | -43.80%       | —                      |
| Risk              | VaR 95% daily (%)                     | -2.10%          | -2.63%        | —                      |
| Risk              | VaR 95% daily (₹, on initial capital) | -₹2,102.66      | -₹2,634.87    | —                      |
| Risk-Adjusted     | Sharpe Ratio                          | 0.82            | 0.98          | —                      |
| Risk-Adjusted     | Sortino Ratio                         | 1.24            | 1.45          | —                      |
| ML Predictive Fit | RMSE (USD)                            | 2.18            | —             | 2.17                   |
| ML Predictive Fit | MAPE (%)                              | 1.24%           | —             | 1.23%                  |

Verdict: INCONCLUSIVE. Gradient diagnostic — across folds the top-layer gradient at the oldest step is 3.80e-11× the most-recent step (strong decay; effective memory ≈9 of 60 steps at a 1% threshold); the input-sensitivity ratio ‖∂L/∂x₁‖/‖∂L/∂x₆₀‖ is 6.94e-06, which is the number to compare against the LSTM run. Crash behaviour (out-of-sample only): 2012–2013 drawdown: strategy -24.4% vs B&H -43.8%, MDD -28.5% vs -43.8%, flat 43% of days (overall 36%) → mixed; 2018–2019 drawdown: strategy -10.8% vs B&H -38.5%, MDD -16.1% vs -38.5%, flat 40% of days (overall 36%) → mixed; 2020 COVID crash: strategy +11.6% vs B&H -31.4%, MDD -16.5% vs -31.4%, flat 41% of days (overall 36%) → mixed; 2022 drawdown: strategy -8.3% vs B&H -30.9%, MDD -18.8% vs -30.9%, flat 62% of days (overall 36%) → mixed; 2024–2025 drawdown: strategy -36.3% vs B&H -33.4%, MDD -36.3% vs -33.4%, flat 38% of days (overall 36%) → failed. 'Supported' requires strong decay AND failure in most out-of-sample crashes; a model that is flat most of the time is not credited with crash protection unless it is flat more often during crashes than overall. Caveat: the 2008 GFC lies inside the in-sample initial-training region, so it cannot count as evidence either way, and decay alone does not prove crash failure — compare with the LSTM run (python main.py --model lstm) before attributing differences to the cell.

Limitations: (1) single asset — AAPL only, so nothing here generalises across stocks or sectors; (2) survivorship/outlier bias — AAPL is one of the best-performing large caps of the last 20 years, so Buy & Hold is an unusually hard benchmark and any long-only result is flattered by the asset choice; (3) regime sparsity — only a handful of macro crashes exist in 20 years, and 2008 falls in the in-sample training region, so crash conclusions rest on very few events; (4) 1-day-ahead price-level targets sit very close to persistence (tomorrow ≈ today), so low RMSE/MAPE mostly reflects that, not skill — compare against the persistence row; (5) the backtest ignores taxes, FX (USD/INR held constant), slippage beyond the flat per-side cost, and market impact.
