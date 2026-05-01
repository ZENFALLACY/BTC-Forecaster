"""Streamlit dashboard for BTCUSDT GBM prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.data_loader import DataLoadError, fetch_btcusdt_klines
from src.gbm_model import GBMConfig, Prediction, predict_next_hour


@st.cache_data(ttl=60, show_spinner=False)
def load_data(limit: int = 720) -> pd.DataFrame:
    return fetch_btcusdt_klines(limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def load_backtest_metrics(
    data: pd.DataFrame,
    volatility_window: int,
    student_t_df: float,
    interval_scale: float,
    use_ewma_volatility: bool,
) -> dict[str, float]:
    config = GBMConfig(
        volatility_window=volatility_window,
        drift_window=volatility_window,
        student_t_df=student_t_df,
        interval_scale=interval_scale,
        use_ewma_volatility=use_ewma_volatility,
        ewma_span=volatility_window,
    )
    _, metrics = run_backtest(data=data, config=config, persist=False)
    return metrics


def target_timestamp(prediction: Prediction) -> pd.Timestamp:
    return prediction.timestamp + pd.Timedelta(hours=1)


def build_prediction_chart(data: pd.DataFrame, prediction: Prediction) -> go.Figure:
    chart_data = data.tail(50)
    last_time = pd.Timestamp(chart_data["close_time"].iloc[-1])
    next_time = target_timestamp(prediction)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["close_time"],
            y=chart_data["close"],
            mode="lines",
            line=dict(color="#111827", width=2.5),
            name="Close price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_time, next_time],
            y=[prediction.predicted_upper, prediction.predicted_upper],
            mode="lines",
            line=dict(color="rgba(30, 136, 229, 0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_time, next_time],
            y=[prediction.predicted_lower, prediction.predicted_lower],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(30, 136, 229, 0.24)",
            line=dict(color="rgba(30, 136, 229, 0)"),
            name="95% prediction range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_time],
            y=[prediction.predicted_median],
            mode="markers",
            marker=dict(color="#1e88e5", size=8),
            name="Median forecast",
        )
    )
    fig.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Close Price With Next-Hour 95% Prediction Range",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_yaxes(title="USDT")
    return fig


def build_candlestick_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.tail(50)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_data["close_time"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="BTCUSDT",
        )
    )
    fig.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Last 50 Closed Hourly Candles",
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title="USDT")
    return fig


def main() -> None:
    st.set_page_config(page_title="BTCUSDT GBM Forecast", layout="wide")
    st.title("BTCUSDT Next-Hour GBM Forecast")

    with st.sidebar:
        st.header("Model settings")
        volatility_window = st.slider("Volatility window", 20, 200, 50, 5)
        student_t_df = st.slider("Student-t df", 3.0, 30.0, 6.0, 0.5)
        interval_scale = st.slider("Interval scale", 0.5, 2.0, 1.12, 0.01)
        use_ewma = st.toggle("EWMA volatility", value=False)

    try:
        data = load_data(limit=800)
    except DataLoadError as exc:
        st.error(str(exc))
        st.stop()

    config = GBMConfig(
        volatility_window=volatility_window,
        drift_window=volatility_window,
        student_t_df=student_t_df,
        interval_scale=interval_scale,
        use_ewma_volatility=use_ewma,
        ewma_span=volatility_window,
    )
    prediction = predict_next_hour(data.tail(500), config)
    metrics = load_backtest_metrics(data, volatility_window, student_t_df, interval_scale, use_ewma)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current BTCUSDT", f"${prediction.current_price:,.2f}")
    c2.metric("Predicted lower", f"${prediction.predicted_lower:,.2f}")
    c3.metric("Predicted upper", f"${prediction.predicted_upper:,.2f}")
    c4.metric("Interval width", f"${prediction.width:,.2f}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Coverage 95", f"{metrics['coverage_95']:.2%}")
    m2.metric("Average width", f"${metrics['average_width']:,.2f}")
    m3.metric("Winkler score", f"{metrics['winkler_score']:,.2f}")

    st.plotly_chart(build_prediction_chart(data, prediction), use_container_width=True)
    st.plotly_chart(build_candlestick_chart(data), use_container_width=True)


if __name__ == "__main__":
    main()
