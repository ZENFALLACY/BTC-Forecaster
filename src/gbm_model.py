"""Geometric Brownian Motion interval model with Student-t innovations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GBMConfig:
    """Model parameters for one-hour-ahead prediction."""

    volatility_window: int = 50
    drift_window: int = 50
    n_paths: int = 10_000
    confidence: float = 0.95
    student_t_df: float = 6.0
    use_ewma_volatility: bool = False
    ewma_span: int = 50
    interval_scale: float = 1.12
    random_seed: int | None = 42
    min_returns: int = 20


@dataclass(frozen=True)
class Prediction:
    """One next-hour forecast interval."""

    timestamp: pd.Timestamp
    current_price: float
    predicted_lower: float
    predicted_upper: float
    predicted_median: float
    drift: float
    volatility: float

    @property
    def width(self) -> float:
        return self.predicted_upper - self.predicted_lower


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute close-to-close log returns."""
    prices = pd.to_numeric(close, errors="coerce")
    return np.log(prices / prices.shift(1)).dropna()


def estimate_drift(log_returns: pd.Series, config: GBMConfig) -> float:
    """Estimate recent one-hour drift from data available at forecast time."""
    sample = log_returns.tail(config.drift_window).dropna()
    return 0.0 if sample.empty else float(sample.mean())


def estimate_volatility(log_returns: pd.Series, config: GBMConfig) -> float:
    """Estimate recent one-hour volatility with optional EWMA weighting."""
    sample = log_returns.tail(config.volatility_window).dropna()
    if len(sample) < 2:
        raise ValueError("At least two log returns are required for volatility")

    if config.use_ewma_volatility:
        # Uses only past squared returns and reacts faster to volatility clusters.
        variance = sample.pow(2).ewm(span=config.ewma_span, adjust=False).mean().iloc[-1]
        return float(np.sqrt(max(variance, 0.0)))
    return float(sample.std(ddof=1))


def simulate_gbm_next_hour(
    current_price: float,
    drift: float,
    volatility: float,
    config: GBMConfig,
) -> np.ndarray:
    """Simulate next-hour prices using standardized Student-t shocks."""
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if config.student_t_df <= 2:
        raise ValueError("student_t_df must be greater than 2 for finite variance")

    rng = np.random.default_rng(config.random_seed)
    # Student-t shocks preserve the GBM structure while allowing fatter tails
    # than a normal distribution, which is important for BTC jumps.
    shocks = rng.standard_t(df=config.student_t_df, size=config.n_paths)
    shocks *= np.sqrt((config.student_t_df - 2.0) / config.student_t_df)
    shocks *= config.interval_scale

    # One-hour GBM step in log-price space.
    next_log_price = np.log(current_price) + drift + volatility * shocks
    return np.exp(next_log_price)


def predict_next_hour(data: pd.DataFrame, config: GBMConfig | None = None) -> Prediction:
    """Predict a 95% interval for the next hourly close using only `data`."""
    cfg = config or GBMConfig()
    if "close" not in data.columns:
        raise ValueError("data must include a 'close' column")

    close = pd.to_numeric(data["close"], errors="coerce").dropna()
    returns = compute_log_returns(close)
    if len(returns) < cfg.min_returns:
        raise ValueError(f"Need at least {cfg.min_returns} log returns, got {len(returns)}")

    current_price = float(close.iloc[-1])
    drift = estimate_drift(returns, cfg)
    volatility = estimate_volatility(returns, cfg)
    simulated_prices = simulate_gbm_next_hour(current_price, drift, volatility, cfg)

    alpha = 1.0 - cfg.confidence
    lower, median, upper = np.quantile(
        simulated_prices,
        [alpha / 2.0, 0.5, 1.0 - alpha / 2.0],
    )

    timestamp_column = "close_time" if "close_time" in data.columns else "open_time"
    timestamp = pd.Timestamp(data[timestamp_column].iloc[-1])
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    return Prediction(
        timestamp=timestamp,
        current_price=current_price,
        predicted_lower=float(lower),
        predicted_upper=float(upper),
        predicted_median=float(median),
        drift=drift,
        volatility=volatility,
    )
