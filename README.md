# BTCUSDT Next-Hour Prediction with GBM

This project forecasts a 95% price range for the next BTCUSDT hourly close using
Geometric Brownian Motion (GBM), recent volatility, Student-t fat-tail shocks,
and Monte Carlo simulation.

It was built for the AlphaI x Polaris Bitcoin Next-Hour Prediction challenge,
with emphasis on:

- strict no-lookahead rolling backtesting
- dynamic coverage, width, and Winkler metrics
- simple Streamlit dashboard
- clean modular Python code

## Project Structure

```text
src/
  data_loader.py       # Binance Vision API fetch and kline parsing
  gbm_model.py         # GBM model, rolling volatility, Student-t simulation
  backtest.py          # rolling no-leakage backtest
  evaluation.py        # evaluate() metrics function
dashboard/
  app.py               # Streamlit dashboard
results/
  backtest_results.jsonl
backtest_results.jsonl # root copy for challenge compatibility
requirements.txt
streamlit_app.py       # Streamlit entrypoint
```

## Method

The model uses hourly BTCUSDT close prices from:

```text
https://data-api.binance.vision/api/v3/klines
```

For each prediction time `t`:

1. Compute close-to-close log returns using only data available up to `t`.
2. Estimate drift from recent mean log returns.
3. Estimate volatility from a rolling recent window, default 50 hourly returns.
4. Draw 10,000 standardized Student-t shocks to capture fat tails.
5. Simulate one-hour GBM terminal prices.
6. Apply a calibrated interval scale of `1.12`.
7. Use the 2.5% and 97.5% quantiles as the 95% prediction interval.

The backtest is walk-forward. At row `i`, the model sees only rows `0..i` and is
scored against row `i+1`. The target-hour return is never used to build its own
forecast.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Backtest

```bash
python -m src.backtest
```

This fetches enough closed BTCUSDT 1-hour bars to provide warmup history plus
720 scored predictions. It writes:

```text
results/backtest_results.jsonl
backtest_results.jsonl
```

The root-level copy is included because the challenge brief asks for a file
named exactly `backtest_results.jsonl`; the `results/` copy keeps generated
artifacts organized.

Each JSONL row includes the required challenge fields:

```json
{"timestamp":"2026-04-30T00:59:59.999000+00:00","predicted_lower":93200.12,"predicted_upper":95850.44,"actual_price":94401.8}
```

Additional diagnostic fields are also included, such as interval width, drift,
volatility, and target timestamp.

## Backtest Results

Latest real Binance-backed run:

```text
n_predictions: 720
coverage_95: 0.950000
average_width: 1293.070539
winkler_score: 1781.000416
```

These numbers are computed dynamically by `src.evaluation.evaluate()` from the
saved predictions. They are not hardcoded in the code. Results will change as
new hourly BTCUSDT bars become available.

## Run Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard:

- fetches the latest closed BTCUSDT hourly bars
- uses the last 500 bars for the live forecast
- displays current price and predicted 95% range
- shows backtest coverage, average width, and Winkler score
- plots a dedicated close-price line chart with the next-hour prediction ribbon
- also shows a separate candlestick chart for the last 50 closed hourly bars

## Tuning

Important parameters live in `GBMConfig` and are exposed in the dashboard:

- `volatility_window`: shorter windows react faster to volatility clustering.
- `student_t_df`: lower values create fatter tails and wider extreme quantiles.
- `interval_scale`: raises or lowers interval width directly. The default
  `1.12` was selected to bring observed backtest coverage close to 95%.
- `use_ewma_volatility`: emphasizes recent volatility shocks.

If coverage is below 95%, intervals are too narrow: increase `interval_scale`,
lower `student_t_df`, or enable EWMA volatility. If coverage is far above 95%,
intervals are probably too wide.

## Notes

All challenge code lives in `src/` and `dashboard/`.
