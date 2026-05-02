"""Streamlit dashboard for BTCUSDT GBM prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.data_loader import DataLoadError, fetch_btcusdt_klines
from src.gbm_model import GBMConfig, Prediction, predict_next_hour
from src.model_comparison import compare_models


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def load_data(limit: int = 720) -> pd.DataFrame:
    return fetch_btcusdt_klines(limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def load_backtest_summary(
    data: pd.DataFrame,
    volatility_window: int,
    student_t_df: float,
    interval_scale: float,
    use_ewma_volatility: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    config = GBMConfig(
        volatility_window=volatility_window,
        drift_window=volatility_window,
        student_t_df=student_t_df,
        interval_scale=interval_scale,
        use_ewma_volatility=use_ewma_volatility,
        ewma_span=volatility_window,
    )
    results, metrics = run_backtest(data=data, config=config, persist=False)
    return results, metrics


@st.cache_data(ttl=600, show_spinner="Running model comparison…")
def load_model_comparison(
    data: pd.DataFrame,
    volatility_window: int,
    student_t_df: float,
    interval_scale: float,
    use_ewma_volatility: bool,
) -> pd.DataFrame:
    config = GBMConfig(
        volatility_window=volatility_window,
        drift_window=volatility_window,
        student_t_df=student_t_df,
        interval_scale=interval_scale,
        use_ewma_volatility=use_ewma_volatility,
        ewma_span=volatility_window,
    )
    return compare_models(data=data, config=config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def target_timestamp(prediction: Prediction) -> pd.Timestamp:
    return prediction.timestamp + pd.Timedelta(hours=1)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


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
            line=dict(color="#00c2ff", width=3.25),
            name="Close price",
            hovertemplate="Time=%{x}<br>Close=$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_time],
            y=[prediction.current_price],
            mode="markers",
            marker=dict(color="#ffb000", size=11, line=dict(color="#111827", width=1)),
            name="Current price",
            hovertemplate="Current close<br>Time=%{x}<br>Price=$%{y:,.2f}<extra></extra>",
        )
    )
    band_color = "rgba(245, 158, 11, 0.28)" if prediction.volatility > 0.01 else "rgba(30, 136, 229, 0.24)"
    fig.add_trace(
        go.Scatter(
            x=[last_time, next_time],
            y=[prediction.predicted_upper, prediction.predicted_upper],
            mode="lines",
            line=dict(color="#1e88e5", width=2, dash="dash"),
            name="Upper bound",
            hovertemplate="Upper 95% bound<br>Time=%{x}<br>Price=$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_time, next_time],
            y=[prediction.predicted_lower, prediction.predicted_lower],
            mode="lines",
            fill="tonexty",
            fillcolor=band_color,
            line=dict(color="#1e88e5", width=2, dash="dash"),
            name="Lower bound",
            hovertemplate="Lower 95% bound<br>Time=%{x}<br>Price=$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_time],
            y=[prediction.predicted_median],
            mode="markers",
            marker=dict(color="#d6336c", size=11, symbol="diamond"),
            name="Median forecast",
            hovertemplate="Forecast hour<br>Time=%{x}<br>Median=$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_time, next_time],
            y=[prediction.predicted_lower, prediction.predicted_upper],
            mode="lines",
            line=dict(color="#d6336c", width=1.5, dash="dot"),
            name="Forecast hour",
            hovertemplate="Forecast target<br>Time=%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Close Price With Next-Hour 95% Prediction Range",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        hovermode="x unified",
    )
    fig.update_xaxes(title="Time", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title="BTCUSDT Price (USDT)", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def build_backtest_chart(results: pd.DataFrame) -> go.Figure:
    """Show recent realized prices against historical forecast intervals.

    Missed predictions (covered == False) are highlighted with large red ✕
    markers to make failures visually prominent.
    """
    chart_data = results.tail(120).copy()
    misses = chart_data[chart_data["covered"] == False]  # noqa: E712

    fig = go.Figure()
    # Prediction band
    fig.add_trace(
        go.Scatter(
            x=chart_data["target_timestamp"],
            y=chart_data["predicted_upper"],
            mode="lines",
            line=dict(color="rgba(34, 139, 230, 0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["target_timestamp"],
            y=chart_data["predicted_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(34, 139, 230, 0.18)",
            line=dict(color="rgba(34, 139, 230, 0)", width=0),
            name="Predicted 95% range",
            hovertemplate="Time=%{x}<br>Lower=$%{y:,.2f}<extra></extra>",
        )
    )
    # Actual price line with coverage-colored dots
    fig.add_trace(
        go.Scatter(
            x=chart_data["target_timestamp"],
            y=chart_data["actual_price"],
            mode="lines+markers",
            line=dict(color="#111827", width=2),
            marker=dict(
                color=chart_data["covered"].map({True: "#2f9e44", False: "#e03131"}),
                size=5,
            ),
            name="Actual close",
            hovertemplate="Time=%{x}<br>Actual=$%{y:,.2f}<extra></extra>",
        )
    )
    # Highlighted miss markers
    if not misses.empty:
        fig.add_trace(
            go.Scatter(
                x=misses["target_timestamp"],
                y=misses["actual_price"],
                mode="markers",
                marker=dict(
                    color="#e03131",
                    size=12,
                    symbol="x",
                    line=dict(color="#e03131", width=2),
                ),
                name="Miss (outside band)",
                hovertemplate="MISS<br>Time=%{x}<br>Actual=$%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Backtest: Actual Close vs Predicted Range",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(title="Target time", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title="BTCUSDT Price (USDT)", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def build_volatility_vs_width_chart(results: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: rolling volatility and prediction interval width over time.

    Demonstrates the relationship: when volatility increases, the model widens
    its prediction interval to maintain target coverage.
    """
    chart_data = results.tail(200).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=chart_data["target_timestamp"],
            y=chart_data["volatility"],
            mode="lines",
            line=dict(color="#7c3aed", width=2),
            name="Volatility (σ)",
            hovertemplate="Time=%{x}<br>σ=%{y:.5f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["target_timestamp"],
            y=chart_data["interval_width"],
            mode="lines",
            line=dict(color="#0ea5e9", width=2),
            name="Interval width ($)",
            hovertemplate="Time=%{x}<br>Width=$%{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=340,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Volatility vs Prediction Interval Width",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        hovermode="x unified",
    )
    fig.update_xaxes(title="Time", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title="Volatility (σ)", showgrid=True, gridcolor="rgba(0,0,0,0.08)", secondary_y=False)
    fig.update_yaxes(title="Interval Width (USDT)", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="rgba(0,0,0,0.08)", secondary_y=True)
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
    fig.update_xaxes(title="Time", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title="BTCUSDT Price (USDT)", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="BTCUSDT GBM Forecast", layout="wide")
    st.title("BTCUSDT Next-Hour GBM Forecast")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model settings")
        volatility_window = st.slider("Volatility window", 20, 200, 50, 5)
        student_t_df = st.slider("Student-t df", 3.0, 30.0, 6.0, 0.5)
        interval_scale = st.slider("Interval scale", 0.5, 2.0, 1.12, 0.01)
        use_ewma = st.toggle("EWMA volatility", value=False)

    # ── Data fetch ───────────────────────────────────────────────────────
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
    backtest_results, metrics = load_backtest_summary(
        data,
        volatility_window,
        student_t_df,
        interval_scale,
        use_ewma,
    )

    # ── SECTION 1: Backtest Metrics ──────────────────────────────────────
    st.header("📊 Backtest Metrics")
    st.caption("Backtested over ~720 hourly bars using strict walk-forward validation with no lookahead bias.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Coverage 95", f"{metrics['coverage_95']:.2%}")
    m2.metric("Average width", f"${metrics['average_width']:,.2f}")
    m3.metric("Winkler score", f"{metrics['winkler_score']:,.2f}")

    # ── SECTION 2: Live Forecast ─────────────────────────────────────────
    st.header("🔮 Live Forecast")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current BTCUSDT", f"${prediction.current_price:,.2f}")
    c2.metric("Predicted lower", f"${prediction.predicted_lower:,.2f}")
    c3.metric("Predicted upper", f"${prediction.predicted_upper:,.2f}")
    c4.metric("Interval width", f"${prediction.width:,.2f}")

    st.plotly_chart(build_prediction_chart(data, prediction), width="stretch")
    st.plotly_chart(build_candlestick_chart(data), width="stretch")

    # ── SECTION 3: Model Comparison ──────────────────────────────────────
    st.header("⚖️ Model Comparison — Student-t vs Normal")
    st.caption(
        "Both models are backtested on the same 720-bar dataset with identical "
        "walk-forward logic. Only the shock distribution differs."
    )
    comparison_df = load_model_comparison(
        data, volatility_window, student_t_df, interval_scale, use_ewma
    )
    st.dataframe(
        comparison_df.style.format(
            {"Coverage": "{:.2%}", "Avg Width": "${:,.2f}", "Winkler": "{:,.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "**Why Student-t wins:** BTC returns exhibit *fat tails* — extreme hourly "
        "moves occur more often than a Normal distribution predicts. The Student-t "
        "distribution captures these heavy tails, producing wider intervals during "
        "volatile periods and achieving higher coverage without excessive average "
        "width. The Normal model underestimates tail risk, leading to lower "
        "coverage and higher Winkler penalties from misses."
    )

    # ── SECTION 4: Insights ──────────────────────────────────────────────
    st.header("💡 Insights")
    with st.expander("Model insights", expanded=True):
        st.markdown(
            "- **Volatility-adaptive intervals:** Prediction intervals widen when "
            "recent BTC volatility rises, and narrow during calm periods. This is "
            "because rolling volatility is the primary driver of range width.\n"
            "- **No static risk estimate:** The model adapts to current market "
            "conditions instead of using one fixed historical volatility.\n"
            "- **Fat-tail awareness:** Student-t shocks give the GBM simulation "
            "heavier tails than Gaussian shocks, reducing underestimation of "
            "extreme BTC moves.\n"
            "- **Calibrated scaling:** The interval scale factor was tuned via "
            "walk-forward backtesting to bring observed coverage close to the "
            "95% target."
        )

    st.subheader("Volatility vs Prediction Width")
    st.caption(
        "During volatile periods, the model increases the prediction range width "
        "to maintain coverage. The chart below shows how these two signals track "
        "each other over the backtest window."
    )
    st.plotly_chart(
        build_volatility_vs_width_chart(backtest_results), width="stretch"
    )

    # ── SECTION 5: Backtest Interpretability ─────────────────────────────
    st.header("🔍 Backtest Interpretability")
    st.plotly_chart(build_backtest_chart(backtest_results), width="stretch")

    n_total = len(backtest_results)
    n_misses = int((~backtest_results["covered"]).sum()) if n_total > 0 else 0
    miss_rate = n_misses / n_total if n_total > 0 else 0.0
    st.caption(
        f"**Miss rate: {miss_rate:.1%}** — {n_misses} of {n_total} predictions "
        f"fell outside the 95% band. Misses typically cluster around sudden "
        f"price spikes or drops, indicating the model's limitation in capturing "
        f"extreme intra-hour momentum."
    )

    # ── SECTION 6: Limitations ───────────────────────────────────────────
    st.header("⚠️ Limitations")
    with st.expander("Known model limitations", expanded=True):
        st.markdown(
            "- **Constant drift assumption:** GBM assumes a stationary drift, "
            "which may not adapt well to regime shifts (e.g. bull-to-bear "
            "transitions).\n"
            "- **Volatility estimation lag:** Rolling volatility is estimated "
            "from recent data and may lag sudden spikes caused by flash crashes "
            "or liquidation cascades.\n"
            "- **No external signals:** The model uses only price history. It "
            "ignores on-chain data, funding rates, news sentiment, and order "
            "flow.\n"
            "- **Extreme event risk:** Even with Student-t tails, truly "
            "unprecedented moves (e.g. exchange hacks, regulatory bans) may "
            "still fall outside the predicted interval.\n"
            "- **Single time horizon:** The model forecasts one hour ahead. "
            "Multi-horizon or intra-hour dynamics are not captured."
        )


if __name__ == "__main__":
    main()
