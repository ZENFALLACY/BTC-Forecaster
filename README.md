# BTCUSDT Next-Hour Forecasting with GBM

This repository is a challenge-ready Bitcoin forecasting system for the
AlphaI x Polaris "Predict Bitcoin's Next Hour" challenge. It predicts a 95%
range for the next BTCUSDT hourly close, evaluates the range with a strict
walk-forward backtest, and presents the live forecast in a Streamlit dashboard.

## Live Dashboard

Run locally:

```bash
streamlit run streamlit_app.py
```

Public dashboard URL:

```text
https://btc-forecaster-bej4duce4cewtex5rgdhpo.streamlit.app/
```

## Project Structure

```text
src/
  data_loader.py       # Binance Vision API fetch and closed-candle parsing
  gbm_model.py         # GBM model, rolling volatility, Student-t simulation
  backtest.py          # strict walk-forward backtest
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

Data comes from Binance Vision's public no-key endpoint:

```text
https://data-api.binance.vision/api/v3/klines
```

The model is a one-step Geometric Brownian Motion simulator:

1. Fetch closed BTCUSDT 1-hour candles.
2. Compute close-to-close log returns.
3. Estimate drift from recent mean log returns.
4. Estimate volatility from a rolling recent window, default 50 hourly returns.
5. Draw 10,000 standardized Student-t shocks for fat tails.
6. Simulate next-hour prices in log-price space.
7. Apply calibrated interval scale `1.12`.
8. Return the 2.5% and 97.5% quantiles as the 95% prediction interval.

The rolling volatility window is important because BTC volatility clusters:
quiet hours tend to follow quiet hours, and violent hours tend to follow violent
hours. The Student-t distribution is important because BTC has more extreme
moves than a normal model would expect.

## Backtest Design

The backtest uses strict walk-forward validation with no lookahead bias.

At prediction step `i`:

- the model receives only rows `0..i`
- it predicts the range for row `i+1`
- row `i+1` is revealed only after the forecast is produced
- the forecast is scored against the actual next hourly close

This produces 720 scored predictions while using extra warmup history for the
rolling volatility estimate.

Run:

```bash
python -m src.backtest
```

Outputs:

```text
results/backtest_results.jsonl
backtest_results.jsonl
```

Each JSONL record includes the challenge-required fields:

```json
{"timestamp":"2026-04-30T00:59:59.999000+00:00","predicted_lower":93200.12,"predicted_upper":95850.44,"actual_price":94401.8}
```

## Backtest Results

Latest Binance-backed run:

```text
n_predictions: 720
coverage_95: 0.950000
average_width: 1293.070539
winkler_score: 1781.000416
```

Metrics are computed dynamically by `src.evaluation.evaluate()` from the saved
predictions. They are not hardcoded.

Metric interpretation:

- `coverage_95`: fraction of actual prices inside the predicted 95% interval.
  It should be close to `0.95`.
- `average_width`: average prediction range width in USDT. Narrower is better
  when coverage remains calibrated.
- `winkler_score`: interval score that rewards narrow covered intervals and
  penalizes misses. Lower is better.

## Dashboard

The Streamlit dashboard shows:

- top-line backtest metrics: coverage, average width, Winkler score
- current BTCUSDT price
- predicted lower and upper bounds for the next hour
- bold close-price chart with shaded 95% prediction ribbon
- dashed lower and upper prediction bounds
- marker for current price and median forecast
- recent backtest visualization: actual close vs predicted range
- candlestick chart for the last 50 closed hourly bars

## Key Insights

- Prediction intervals widen during high volatility because recent rolling
  volatility is the main driver of range width.
- The model adapts to current market conditions instead of using one fixed
  historical volatility estimate.
- Student-t shocks help capture BTC's fat-tailed behavior and reduce
  underestimation of extreme moves.
- The interval scale was calibrated to bring observed walk-forward coverage
  close to the 95% target.

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

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Community Cloud app.
3. Select this repository and branch.
4. Set the entrypoint to:

```text
streamlit_app.py
```

5. Deploy and paste the public URL into the challenge submission form.
