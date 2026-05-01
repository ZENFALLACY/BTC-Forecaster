"""Binance Vision API data fetching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
import requests


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


class BinanceDataError(RuntimeError):
    """Raised when Binance data cannot be fetched or parsed."""


@dataclass(frozen=True)
class KlineRequest:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 720
    timeout: float = 15.0


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


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 720,
    timeout: float = 15.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch Binance klines and return a clean OHLCV dataframe.

    Binance returns the most recent closed and possibly in-progress candle. The
    model can use the latest available close for live prediction; backtests rely
    only on historical rows already present in the dataframe.
    """
    params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    client = session or requests.Session()

    try:
        response = client.get(BINANCE_KLINES_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise BinanceDataError(f"Failed to fetch Binance klines: {exc}") from exc
    except ValueError as exc:
        raise BinanceDataError("Binance response was not valid JSON") from exc

    if not isinstance(payload, list) or not payload:
        raise BinanceDataError("Binance returned no kline data")

    return parse_klines(payload)


def parse_klines(payload: list[list[Any]]) -> pd.DataFrame:
    """Convert raw Binance kline payload into typed OHLCV rows."""
    frame = pd.DataFrame(payload, columns=KLINE_COLUMNS)

    numeric_cols = [
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
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.drop(columns=["ignore"]).dropna(subset=["open", "high", "low", "close"])
    frame = frame.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return frame


def add_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of data with close-to-close log returns."""
    frame = data.copy()
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame
