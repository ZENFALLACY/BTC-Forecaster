"""Prediction interval evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winkler_score(
    actual: float,
    lower: float,
    upper: float,
    alpha: float = 0.05,
) -> float:
    """Compute Winkler interval score for one observation.

    Lower is better. The score rewards narrow intervals but applies a large
    penalty when the actual value falls outside the interval.
    """
    width = upper - lower
    if actual < lower:
        return float(width + (2.0 / alpha) * (lower - actual))
    if actual > upper:
        return float(width + (2.0 / alpha) * (actual - upper))
    return float(width)


def evaluate_intervals(results: pd.DataFrame, confidence: float = 0.95) -> dict[str, float]:
    """Evaluate interval forecasts from a dataframe of lower/upper/actual rows."""
    required = {"lower", "upper", "actual"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if results.empty:
        return {
            "n_predictions": 0,
            "coverage_95": np.nan,
            "avg_width": np.nan,
            "winkler_score": np.nan,
        }

    alpha = 1.0 - confidence
    frame = results.copy()
    frame["covered"] = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
    frame["width"] = frame["upper"] - frame["lower"]
    frame["winkler"] = [
        winkler_score(row.actual, row.lower, row.upper, alpha=alpha)
        for row in frame.itertuples(index=False)
    ]

    return {
        "n_predictions": int(len(frame)),
        "coverage_95": float(frame["covered"].mean()),
        "avg_width": float(frame["width"].mean()),
        "winkler_score": float(frame["winkler"].mean()),
    }
