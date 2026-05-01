"""BTCUSDT hourly data loading from Binance Vision."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import requests


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class DataLoadError(RuntimeError):
    """Raised when Binance data cannot be loaded."""


def fetch_btcusdt_klines(
    limit: int = 720,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch hourly BTCUSDT bars from Binance Vision."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise DataLoadError(f"Binance request failed: {exc}") from exc
    except ValueError as exc:
        raise DataLoadError("Binance returned invalid JSON") from exc

    if not isinstance(payload, list) or not payload:
        raise DataLoadError("Binance returned no kline rows")
    return parse_klines(payload)


def parse_klines(payload: list[list[Any]]) -> pd.DataFrame:
    """Normalize Binance kline rows into typed, ascending OHLCV data."""
    frame = pd.DataFrame(payload, columns=KLINE_COLUMNS)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.drop(columns=["ignore"])
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return frame


def add_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with close-to-close log returns."""
    frame = data.copy()
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame
