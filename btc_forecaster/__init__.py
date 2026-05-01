"""BTC next-hour prediction package."""

from .model import GBMConfig, PredictionInterval, predict_interval

__all__ = ["GBMConfig", "PredictionInterval", "predict_interval"]
