"""Walk-forward backtest for next-hour BTC prediction intervals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import fetch_klines
from .evaluate import evaluate_intervals, winkler_score
from .model import GBMConfig, predict_interval
from .utils import write_jsonl


DEFAULT_BACKTEST_PATH = Path("backtest_results.jsonl")


def run_backtest(
    data: pd.DataFrame | None = None,
    config: GBMConfig | None = None,
    output_path: str | Path = DEFAULT_BACKTEST_PATH,
    min_train_bars: int | None = None,
    persist: bool = True,
) -> dict[str, float]:
    """Run a no-lookahead walk-forward backtest.

    At index i, the model sees rows 0..i only and predicts close[i + 1].
    No realized return from i to i+1 is used during calibration.
    """
    cfg = config or GBMConfig()
    frame = data.copy() if data is not None else fetch_klines(limit=720)
    frame = frame.sort_values("open_time").reset_index(drop=True)

    train_bars = min_train_bars or max(cfg.vol_window + 1, cfg.min_observations + 1)
    if len(frame) <= train_bars:
        raise ValueError(f"Need more than {train_bars} bars for backtest, got {len(frame)}")

    records: list[dict[str, object]] = []
    alpha = 1.0 - cfg.confidence

    for i in range(train_bars - 1, len(frame) - 1):
        history = frame.iloc[: i + 1]
        actual_row = frame.iloc[i + 1]
        prediction = predict_interval(history, cfg)

        lower = prediction.lower
        upper = prediction.upper
        actual = float(actual_row["close"])
        covered = lower <= actual <= upper
        score = winkler_score(actual, lower, upper, alpha=alpha)

        records.append(
            {
                "timestamp": prediction.timestamp.isoformat(),
                "target_timestamp": pd.Timestamp(actual_row["close_time"]).isoformat(),
                "lower": lower,
                "upper": upper,
                "actual": actual,
                "covered": covered,
                "width": upper - lower,
                "winkler": score,
                "current_price": prediction.current_price,
                "drift": prediction.drift,
                "volatility": prediction.volatility,
            }
        )

    if persist:
        write_jsonl(output_path, records)

    metrics = evaluate_intervals(pd.DataFrame(records), confidence=cfg.confidence)
    metrics["target_coverage"] = cfg.confidence
    return metrics


def load_backtest_results(path: str | Path = DEFAULT_BACKTEST_PATH) -> pd.DataFrame:
    """Load persisted backtest JSONL records into a dataframe."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_json(file_path, lines=True)


def main() -> None:
    metrics = run_backtest()
    print("Backtest metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
