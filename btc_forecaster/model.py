"""GBM next-hour interval model using fat-tailed Student-t shocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GBMConfig:
    """Configuration for one-step GBM Monte Carlo prediction."""

    vol_window: int = 50
    drift_window: int | None = 50
    n_paths: int = 10_000
    confidence: float = 0.95
    student_t_df: float = 6.0
    use_ewma_vol: bool = False
    ewma_span: int = 50
    random_seed: int | None = 42
    min_observations: int = 20
    interval_scale: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionInterval:
    """One-hour-ahead prediction interval."""

    timestamp: pd.Timestamp
    current_price: float
    lower: float
    upper: float
    median: float
    drift: float
    volatility: float
    confidence: float
    n_paths: int
    student_t_df: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_record(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "lower": self.lower,
            "upper": self.upper,
            "median": self.median,
            "width": self.width,
            "drift": self.drift,
            "volatility": self.volatility,
            "confidence": self.confidence,
            "n_paths": self.n_paths,
            "student_t_df": self.student_t_df,
        }


def log_returns(close: pd.Series) -> pd.Series:
    """Compute close-to-close log returns."""
    close = pd.to_numeric(close, errors="coerce")
    return np.log(close / close.shift(1)).dropna()


def estimate_drift(returns: pd.Series, config: GBMConfig) -> float:
    """Estimate one-hour drift from historical log returns available at time t."""
    window = config.drift_window or len(returns)
    sample = returns.tail(window).dropna()
    if sample.empty:
        return 0.0
    return float(sample.mean())


def estimate_volatility(returns: pd.Series, config: GBMConfig) -> float:
    """Estimate recent one-hour volatility without using future returns."""
    sample = returns.tail(config.vol_window).dropna()
    if len(sample) < 2:
        raise ValueError("At least two returns are required to estimate volatility")

    if config.use_ewma_vol:
        # EWMA reacts faster to volatility clustering while still using only past returns.
        variance = sample.pow(2).ewm(span=config.ewma_span, adjust=False).mean().iloc[-1]
        return float(np.sqrt(max(variance, 0.0)))

    return float(sample.std(ddof=1))


def simulate_next_price(
    current_price: float,
    drift: float,
    volatility: float,
    config: GBMConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one-hour-ahead GBM terminal prices with Student-t innovations."""
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if config.student_t_df <= 2:
        raise ValueError("student_t_df must be > 2 for finite variance")

    generator = rng or np.random.default_rng(config.random_seed)
    shocks = generator.standard_t(df=config.student_t_df, size=config.n_paths)
    # Standardize Student-t to unit variance so volatility keeps its usual scale.
    shocks *= np.sqrt((config.student_t_df - 2.0) / config.student_t_df)
    shocks *= config.interval_scale
    next_log_price = np.log(current_price) + drift + volatility * shocks
    return np.exp(next_log_price)


def predict_interval(data: pd.DataFrame, config: GBMConfig | None = None) -> PredictionInterval:
    """Predict the next-hour BTC price interval using only rows in data."""
    cfg = config or GBMConfig()
    if len(data) < cfg.min_observations + 1:
        raise ValueError(
            f"Need at least {cfg.min_observations + 1} price rows, got {len(data)}"
        )
    if "close" not in data.columns:
        raise ValueError("data must contain a 'close' column")

    close = pd.to_numeric(data["close"], errors="coerce").dropna()
    returns = log_returns(close)
    if len(returns) < cfg.min_observations:
        raise ValueError(
            f"Need at least {cfg.min_observations} returns, got {len(returns)}"
        )

    drift = estimate_drift(returns, cfg)
    volatility = estimate_volatility(returns, cfg)
    current_price = float(close.iloc[-1])

    seed = cfg.random_seed
    rng = np.random.default_rng(seed)
    paths = simulate_next_price(current_price, drift, volatility, cfg, rng=rng)
    alpha = 1.0 - cfg.confidence
    lower, median, upper = np.quantile(paths, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])

    timestamp_col = "close_time" if "close_time" in data.columns else "open_time"
    timestamp = pd.Timestamp(data[timestamp_col].iloc[-1])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return PredictionInterval(
        timestamp=timestamp,
        current_price=current_price,
        lower=float(lower),
        upper=float(upper),
        median=float(median),
        drift=drift,
        volatility=volatility,
        confidence=cfg.confidence,
        n_paths=cfg.n_paths,
        student_t_df=cfg.student_t_df,
    )


def tune_parameter_grid(
    data: pd.DataFrame,
    vol_windows: list[int],
    dfs: list[float],
    scales: list[float],
    backtest_fn,
) -> pd.DataFrame:
    """Simple hook for grid-searching interval calibration parameters.

    The caller supplies backtest_fn(data, config) to avoid coupling the model
    module to the backtest module. Prefer tuning on a validation period distinct
    from any final reporting period.
    """
    rows: list[dict[str, float]] = []
    for vol_window in vol_windows:
        for df in dfs:
            for scale in scales:
                cfg = GBMConfig(vol_window=vol_window, student_t_df=df, interval_scale=scale)
                metrics = backtest_fn(data, cfg)
                rows.append({"vol_window": vol_window, "student_t_df": df, "scale": scale, **metrics})
    return pd.DataFrame(rows).sort_values(["winkler_score", "avg_width"])
