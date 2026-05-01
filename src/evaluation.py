"""Evaluation metrics for prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winkler_interval_score(
    actual_price: float,
    predicted_lower: float,
    predicted_upper: float,
    alpha: float = 0.05,
) -> float:
    """Compute the Winkler score for one prediction interval.

    Covered observations score the interval width. Misses receive an additional
    distance penalty, so lower scores reward intervals that are both tight and
    well calibrated.
    """
    width = predicted_upper - predicted_lower
    if actual_price < predicted_lower:
        return float(width + (2.0 / alpha) * (predicted_lower - actual_price))
    if actual_price > predicted_upper:
        return float(width + (2.0 / alpha) * (actual_price - predicted_upper))
    return float(width)


def evaluate(predictions: pd.DataFrame, confidence: float = 0.95) -> dict[str, float]:
    """Compute coverage, average width, and Winkler score dynamically.

    coverage_95 should be near 0.95 for a calibrated 95% interval. Values below
    0.95 suggest overconfidence; values far above 0.95 suggest intervals are too
    wide to be useful.
    """
    required = {"predicted_lower", "predicted_upper", "actual_price"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    if predictions.empty:
        return {
            "n_predictions": 0,
            "coverage_95": np.nan,
            "average_width": np.nan,
            "winkler_score": np.nan,
        }

    alpha = 1.0 - confidence
    frame = predictions.copy()
    frame["covered"] = (
        (frame["actual_price"] >= frame["predicted_lower"])
        & (frame["actual_price"] <= frame["predicted_upper"])
    )
    frame["width"] = frame["predicted_upper"] - frame["predicted_lower"]
    frame["winkler"] = [
        winkler_interval_score(
            actual_price=row.actual_price,
            predicted_lower=row.predicted_lower,
            predicted_upper=row.predicted_upper,
            alpha=alpha,
        )
        for row in frame.itertuples(index=False)
    ]

    return {
        "n_predictions": int(len(frame)),
        "coverage_95": float(frame["covered"].mean()),
        "average_width": float(frame["width"].mean()),
        "winkler_score": float(frame["winkler"].mean()),
    }
