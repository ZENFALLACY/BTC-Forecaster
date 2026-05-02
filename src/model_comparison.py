"""Model comparison: Student-t vs Normal GBM on the same backtest data."""

from __future__ import annotations

import pandas as pd

from src.evaluation import evaluate
from src.gbm_model import GBMConfig, predict_next_hour, predict_next_hour_normal


def _run_walk_forward(
    data: pd.DataFrame,
    config: GBMConfig,
    predict_fn,
    target_predictions: int = 720,
) -> dict[str, float]:
    """Run a walk-forward backtest using *predict_fn* and return metrics.

    Reuses the same no-peeking logic as ``src.backtest.run_backtest`` but
    accepts an arbitrary prediction function so we can swap Student-t for
    Normal without duplicating the backtest harness.
    """
    cfg = config
    train_bars = max(cfg.volatility_window + 1, cfg.min_returns + 1)
    frame = data.copy().sort_values("open_time").reset_index(drop=True)

    if len(frame) <= train_bars:
        return {
            "n_predictions": 0,
            "coverage_95": float("nan"),
            "average_width": float("nan"),
            "winkler_score": float("nan"),
        }

    records: list[dict[str, object]] = []
    first_prediction_index = max(train_bars - 1, len(frame) - target_predictions - 1)

    for i in range(first_prediction_index, len(frame) - 1):
        history = frame.iloc[: i + 1]
        target = frame.iloc[i + 1]
        forecast = predict_fn(history, cfg)

        actual_price = float(target["close"])
        records.append(
            {
                "predicted_lower": forecast.predicted_lower,
                "predicted_upper": forecast.predicted_upper,
                "actual_price": actual_price,
            }
        )

    results = pd.DataFrame(records)
    return evaluate(results, confidence=cfg.confidence)


def compare_models(
    data: pd.DataFrame,
    config: GBMConfig | None = None,
    target_predictions: int = 720,
) -> pd.DataFrame:
    """Run Student-t and Normal GBM backtests on *data* and return a comparison.

    Both models use the exact same dataset, config, and walk-forward logic.
    Only the shock distribution differs.

    Returns a DataFrame with columns: Model, Coverage, Avg Width, Winkler.
    """
    cfg = config or GBMConfig()

    student_t_metrics = _run_walk_forward(
        data, cfg, predict_next_hour, target_predictions
    )
    normal_metrics = _run_walk_forward(
        data, cfg, predict_next_hour_normal, target_predictions
    )

    rows = [
        {
            "Model": f"Student-t (df={cfg.student_t_df:.0f})",
            "Coverage": student_t_metrics["coverage_95"],
            "Avg Width": student_t_metrics["average_width"],
            "Winkler": student_t_metrics["winkler_score"],
        },
        {
            "Model": "Normal (Gaussian)",
            "Coverage": normal_metrics["coverage_95"],
            "Avg Width": normal_metrics["average_width"],
            "Winkler": normal_metrics["winkler_score"],
        },
    ]
    return pd.DataFrame(rows)
