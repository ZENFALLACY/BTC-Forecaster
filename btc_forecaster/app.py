"""Streamlit dashboard for BTCUSDT next-hour GBM prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest import run_backtest
from .data import BinanceDataError, fetch_klines
from .model import GBMConfig, PredictionInterval, predict_interval
from .utils import atomic_write_jsonl, read_jsonl


PREDICTION_HISTORY_PATH = Path("prediction_history.jsonl")


def prediction_target_timestamp(prediction: PredictionInterval) -> pd.Timestamp:
    """The next hourly close timestamp for a one-step-ahead prediction."""
    return prediction.timestamp + pd.Timedelta(hours=1)


def update_prediction_history(
    prediction: PredictionInterval,
    market_data: pd.DataFrame,
    path: str | Path = PREDICTION_HISTORY_PATH,
) -> list[dict[str, object]]:
    """Persist the latest prediction and fill actuals for matured predictions."""
    records = read_jsonl(path)
    latest_record = prediction.to_record()
    latest_record["target_timestamp"] = prediction_target_timestamp(prediction).isoformat()
    latest_record["actual"] = None
    latest_record["covered"] = None

    by_timestamp = {record["timestamp"]: record for record in records}
    by_timestamp[latest_record["timestamp"]] = {**by_timestamp.get(latest_record["timestamp"], {}), **latest_record}

    close_lookup = {
        pd.Timestamp(row.close_time).isoformat(): float(row.close)
        for row in market_data.itertuples(index=False)
        if hasattr(row, "close_time")
    }

    updated = []
    for record in sorted(by_timestamp.values(), key=lambda item: item["timestamp"]):
        target = record.get("target_timestamp")
        if target in close_lookup:
            actual = close_lookup[target]
            record["actual"] = actual
            record["covered"] = bool(record["lower"] <= actual <= record["upper"])
        updated.append(record)

    atomic_write_jsonl(path, updated)
    return updated


@st.cache_data(ttl=60, show_spinner=False)
def load_market_data(limit: int) -> pd.DataFrame:
    return fetch_klines(limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def cached_backtest(
    data: pd.DataFrame,
    vol_window: int,
    student_t_df: float,
    use_ewma_vol: bool,
    interval_scale: float,
) -> dict[str, float]:
    cfg = GBMConfig(
        vol_window=vol_window,
        drift_window=vol_window,
        student_t_df=student_t_df,
        use_ewma_vol=use_ewma_vol,
        ewma_span=vol_window,
        interval_scale=interval_scale,
    )
    return run_backtest(data=data, config=cfg, persist=False)


def build_price_chart(data: pd.DataFrame, prediction: PredictionInterval) -> go.Figure:
    chart_data = data.tail(50)
    last_ts = pd.Timestamp(chart_data["close_time"].iloc[-1])
    target_ts = prediction_target_timestamp(prediction)

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
    fig.add_trace(
        go.Scatter(
            x=[last_ts, target_ts, target_ts, last_ts],
            y=[prediction.lower, prediction.lower, prediction.upper, prediction.upper],
            fill="toself",
            fillcolor="rgba(34, 139, 230, 0.18)",
            line=dict(color="rgba(34, 139, 230, 0.0)"),
            name="95% range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target_ts],
            y=[prediction.median],
            mode="markers",
            marker=dict(size=9, color="#1c7ed6"),
            name="Median forecast",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title="USDT")
    return fig


def render_metrics(prediction: PredictionInterval, metrics: dict[str, float]) -> None:
    cols = st.columns(4)
    cols[0].metric("BTCUSDT", f"${prediction.current_price:,.2f}")
    cols[1].metric("95% lower", f"${prediction.lower:,.2f}")
    cols[2].metric("95% upper", f"${prediction.upper:,.2f}")
    cols[3].metric("Width", f"${prediction.width:,.2f}")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Backtest coverage", f"{metrics['coverage_95']:.2%}")
    metric_cols[1].metric("Avg width", f"${metrics['avg_width']:,.2f}")
    metric_cols[2].metric("Winkler score", f"{metrics['winkler_score']:,.2f}")


def main() -> None:
    st.set_page_config(page_title="BTC Next-Hour GBM", page_icon="₿", layout="wide")
    st.title("BTCUSDT Next-Hour Prediction")

    with st.sidebar:
        st.header("Model")
        vol_window = st.slider("Volatility window", min_value=20, max_value=200, value=50, step=5)
        student_t_df = st.slider("Student-t df", min_value=3.0, max_value=30.0, value=6.0, step=0.5)
        interval_scale = st.slider("Interval scale", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
        use_ewma_vol = st.toggle("EWMA volatility", value=False)
        persist_live = st.toggle("Persist live prediction", value=True)

    try:
        data = load_market_data(limit=720)
    except BinanceDataError as exc:
        st.error(str(exc))
        st.stop()

    config = GBMConfig(
        vol_window=vol_window,
        drift_window=vol_window,
        student_t_df=student_t_df,
        use_ewma_vol=use_ewma_vol,
        ewma_span=vol_window,
        interval_scale=interval_scale,
    )
    prediction = predict_interval(data.tail(500), config)
    metrics = cached_backtest(data, vol_window, student_t_df, use_ewma_vol, interval_scale)

    render_metrics(prediction, metrics)
    st.plotly_chart(build_price_chart(data, prediction), use_container_width=True)

    st.subheader("Prediction History")
    if persist_live:
        history = update_prediction_history(prediction, data)
    else:
        history = read_jsonl(PREDICTION_HISTORY_PATH)

    if history:
        history_frame = pd.DataFrame(history).tail(25).sort_values("timestamp", ascending=False)
        st.dataframe(
            history_frame[
                ["timestamp", "target_timestamp", "lower", "upper", "actual", "covered", "width"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No persisted predictions yet.")

    with st.expander("Tuning notes"):
        st.write(
            "Coverage below 95% usually means intervals are too narrow: raise interval scale, "
            "lower Student-t df, shorten the volatility window, or enable EWMA. Coverage far above "
            "95% usually means intervals are too wide: lower interval scale or lengthen the window."
        )


if __name__ == "__main__":
    main()
