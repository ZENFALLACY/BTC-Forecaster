"""Rolling no-leakage backtest for BTCUSDT next-hour intervals."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_loader import fetch_btcusdt_klines
from src.evaluation import evaluate
from src.gbm_model import GBMConfig, predict_next_hour


DEFAULT_RESULTS_PATH = Path("results/backtest_results.jsonl")
ROOT_RESULTS_PATH = Path("backtest_results.jsonl")
DEFAULT_TARGET_PREDICTIONS = 720


def write_jsonl(path: str | Path, records: list[dict[str, object]]) -> None:
    """Write records to JSONL, creating the parent directory if needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")


def run_backtest(
    data: pd.DataFrame | None = None,
    config: GBMConfig | None = None,
    output_path: str | Path = DEFAULT_RESULTS_PATH,
    root_output_path: str | Path | None = ROOT_RESULTS_PATH,
    min_train_bars: int | None = None,
    target_predictions: int = DEFAULT_TARGET_PREDICTIONS,
    persist: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a rolling backtest over roughly 720 scored hourly predictions.

    No-peeking rule:
    at index i, the model receives only rows 0..i. It predicts the next hourly
    close, then the backtest reveals row i+1 only for scoring. The realized
    target return is never available during calibration.
    """
    cfg = config or GBMConfig()
    train_bars = min_train_bars or max(cfg.volatility_window + 1, cfg.min_returns + 1)
    fetch_limit = target_predictions + train_bars
    frame = data.copy() if data is not None else fetch_btcusdt_klines(limit=fetch_limit)
    frame = frame.sort_values("open_time").reset_index(drop=True)

    if len(frame) <= train_bars:
        raise ValueError(f"Need more than {train_bars} bars, got {len(frame)}")

    records: list[dict[str, object]] = []
    first_prediction_index = max(train_bars - 1, len(frame) - target_predictions - 1)
    for i in range(first_prediction_index, len(frame) - 1):
        # Strict walk-forward slice: this is the only data the model can see.
        history = frame.iloc[: i + 1]
        # The next row is held out until after the forecast is produced.
        target = frame.iloc[i + 1]
        forecast = predict_next_hour(history, cfg)

        actual_price = float(target["close"])
        records.append(
            {
                "timestamp": forecast.timestamp.isoformat(),
                "target_timestamp": pd.Timestamp(target["close_time"]).isoformat(),
                "predicted_lower": forecast.predicted_lower,
                "predicted_upper": forecast.predicted_upper,
                "actual_price": actual_price,
                "covered": bool(forecast.predicted_lower <= actual_price <= forecast.predicted_upper),
                "current_price": forecast.current_price,
                "interval_width": forecast.width,
                "drift": forecast.drift,
                "volatility": forecast.volatility,
            }
        )

    results = pd.DataFrame(records)
    metrics = evaluate(results, confidence=cfg.confidence)
    if persist:
        write_jsonl(output_path, records)
        if root_output_path is not None and Path(root_output_path) != Path(output_path):
            write_jsonl(root_output_path, records)
    return results, metrics


def load_backtest_results(path: str | Path = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load persisted backtest results."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_json(file_path, lines=True)


def main() -> None:
    _, metrics = run_backtest()
    print("Backtest metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
