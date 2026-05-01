# Bitcoin Next-Hour Prediction (GBM)

Production-style Python implementation for forecasting a 95% one-hour-ahead
BTCUSDT price interval with Geometric Brownian Motion, Monte Carlo simulation,
rolling volatility, and Student-t innovations.

## What It Does

- Fetches hourly BTCUSDT klines from Binance Vision:
  `https://data-api.binance.vision/api/v3/klines`
- Estimates log-return drift and recent volatility using only data available at
  prediction time.
- Simulates 10,000 one-hour GBM paths with standardized Student-t shocks.
- Produces 2.5% and 97.5% quantiles as the 95% prediction range.
- Runs a strict walk-forward backtest and writes `backtest_results.jsonl`.
- Serves a Streamlit dashboard with current forecast, backtest metrics, chart,
  and optional live prediction persistence.

## Project Structure

```text
btc_forecaster/
├── data.py          # Binance API fetching and kline parsing
├── model.py         # GBM calibration, Student-t simulation, tuning hooks
├── backtest.py      # no-lookahead walk-forward backtest
├── evaluate.py      # coverage, width, Winkler score
├── app.py           # Streamlit dashboard
├── utils.py         # JSONL and timestamp helpers
├── __init__.py
requirements.txt
streamlit_app.py     # Streamlit Cloud-friendly entrypoint
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run Backtest

```bash
python -m btc_forecaster.backtest
```

This fetches about 720 hourly bars, walks forward one hour at a time, and saves
records like:

```json
{"timestamp":"2026-04-29T23:59:59.999000+00:00","lower":92850.12,"upper":96210.44,"actual":94401.80,"covered":true}
```

Example terminal output:

```text
Backtest metrics
n_predictions: 669
coverage_95: 0.947683
avg_width: 3188.421503
winkler_score: 4210.667321
target_coverage: 0.950000
```

Actual values will change with market conditions and tuning parameters.

## Run Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard fetches recent Binance data, computes the current next-hour range,
runs a cached backtest, and optionally writes `prediction_history.jsonl`.

## Streamlit Deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Set the entrypoint to `streamlit_app.py`.
4. Ensure `requirements.txt` is included.
5. Deploy.

## Tuning Notes

Key parameters are exposed in `GBMConfig` and the dashboard sidebar:

- `vol_window`: shorter windows react faster to volatility clustering; longer
  windows are smoother but can underreact.
- `use_ewma_vol`: emphasizes recent large moves and often improves coverage in
  clustered volatility regimes.
- `student_t_df`: lower degrees of freedom create fatter tails and wider tail
  quantiles.
- `interval_scale`: simple calibration multiplier. Increase it if coverage is
  below 95%; decrease it if coverage is far above 95%.

For honest model selection, tune parameters on one validation period and report
final metrics on a separate period.

## No-Lookahead Design

The backtest loop calls `predict_interval(data.iloc[:i+1])` and only then scores
against `close[i+1]`. Rolling volatility and drift are calculated from historical
returns inside that sliced dataframe, so the realized target-hour return is never
used for its own prediction.
