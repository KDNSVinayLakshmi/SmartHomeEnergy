"""Streamlit dashboard for household electricity forecasting and insights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preprocessing import load_consumption_series
from src.forecast import forecast_next_hour, load_model_and_scaler
from src.utils import MODELS_DIR

METRICS_PATH = MODELS_DIR / "metrics.json"


@st.cache_data(show_spinner="Loading household power readings...")
def get_consumption_series() -> pd.Series:
    """Cache the cleaned 15-minute consumption series for the session."""
    return load_consumption_series()


@st.cache_resource(show_spinner="Loading the trained LSTM model...")
def get_model_and_scaler() -> tuple[Any, Any]:
    """Cache model artifacts instead of reloading them on each rerun."""
    return load_model_and_scaler()


def load_metrics(metrics_path: Path = METRICS_PATH) -> dict[str, float] | None:
    """Read test metrics written by training, if available."""
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read evaluation metrics at {metrics_path}: {exc}"
        ) from exc

    required = ("mae", "rmse", "r2")
    if not all(key in metrics for key in required):
        raise ValueError(f"Evaluation metrics file {metrics_path} is incomplete.")
    return {key: float(metrics[key]) for key in required}


def build_recommendations(predicted_power_kw: float, threshold_kw: float) -> list[str]:
    """Choose concise energy-saving advice based on forecast and threshold."""
    if predicted_power_kw > threshold_kw:
        return [
            "Turn off lights and devices that are not in use.",
            "Delay high-load appliances such as laundry or the dishwasher.",
            "Optimize air-conditioning settings and avoid cooling empty rooms.",
        ]
    return ["Maintain efficient consumption and switch off idle devices."]


def main() -> None:
    """Render the dashboard and its forecast-driven household insights."""
    st.set_page_config(
        page_title="Smart Home Energy Forecasting",
        page_icon="🏠",
        layout="wide",
    )
    st.title("🏠 Smart Home Energy Forecasting")
    st.caption(
        "15-minute household power forecasts from the previous 24 hours "
        "of electricity readings."
    )

    st.sidebar.header("Dashboard controls")
    threshold_kw = st.sidebar.slider(
        "High-consumption threshold (kW)",
        min_value=1.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
    )
    history_days = st.sidebar.slider(
        "Historical chart window (days)",
        min_value=1,
        max_value=30,
        value=7,
    )
    horizon_minutes = st.sidebar.selectbox(
        "Forecast horizon",
        options=[15, 30, 45, 60],
        index=0,
        format_func=lambda minutes: f"{minutes} minutes",
    )

    try:
        series = get_consumption_series()
        model, scaler = get_model_and_scaler()
        forecast = forecast_next_hour(series.iloc[-96:], model=model, scaler=scaler)
        metrics = load_metrics()
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        st.error(str(exc))
        st.info(
            "Install the project requirements and train the model from the "
            "project directory with `python -m src.train_model`. The first "
            "run also downloads the UCI dataset."
        )
        st.stop()

    selected = forecast.loc[forecast["minutes_ahead"] == horizon_minutes]
    if selected.empty:
        st.error(
            f"No prediction is available for the {horizon_minutes}-minute horizon."
        )
        st.stop()
    predicted_power_kw = float(selected["predicted_power_kw"].iloc[0])
    current_power_kw = float(series.iloc[-1])
    latest_day = series.index[-1].normalize()
    daily_energy_kwh = float(
        series.loc[series.index.normalize() == latest_day].sum() * 0.25
    )
    peak_usage_kw = float(series.loc[series.index.normalize() == latest_day].max())

    metric_columns = st.columns(5)
    metric_columns[0].metric("Current consumption", f"{current_power_kw:.2f} kW")
    metric_columns[1].metric(
        f"Predicted in {horizon_minutes} min", f"{predicted_power_kw:.2f} kW"
    )
    metric_columns[2].metric("Today's energy", f"{daily_energy_kwh:.2f} kWh")
    metric_columns[3].metric("Today's peak usage", f"{peak_usage_kw:.2f} kW")
    metric_columns[4].metric("Forecast horizon", f"{horizon_minutes} min")

    if predicted_power_kw > threshold_kw:
        st.warning(
            f"⚠ High Power Consumption Expected: "
            f"{predicted_power_kw:.2f} kW is above your "
            f"{threshold_kw:.1f} kW threshold."
        )
    else:
        st.success(
            f"Forecast is within your {threshold_kw:.1f} kW consumption threshold."
        )

    st.subheader("Energy-saving recommendations")
    for recommendation in build_recommendations(predicted_power_kw, threshold_kw):
        st.write(f"- {recommendation}")

    if metrics is None:
        st.info(
            "Model evaluation metrics are not available. "
            "Retrain the model to save them."
        )
    else:
        st.subheader("Model evaluation — chronological test set")
        evaluation_columns = st.columns(3)
        evaluation_columns[0].metric("MAE", f"{metrics['mae']:.3f} kW")
        evaluation_columns[1].metric("RMSE", f"{metrics['rmse']:.3f} kW")
        evaluation_columns[2].metric("R²", f"{metrics['r2']:.3f}")

    st.subheader("Historical consumption")
    history_start = series.index[-1] - pd.Timedelta(days=history_days)
    history = (
        series.loc[series.index >= history_start]
        .rename("Power (kW)")
        .reset_index()
    )
    history.columns = ["Time", "Power (kW)"]
    history_chart = px.line(
        history,
        x="Time",
        y="Power (kW)",
        title=f"Last {history_days} days",
        labels={"Time": "Time", "Power (kW)": "Power (kW)"},
    )
    history_chart.update_layout(hovermode="x unified")
    st.plotly_chart(history_chart, use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Next-hour forecast")
        forecast_times = series.index[-1] + pd.to_timedelta(
            forecast["minutes_ahead"], unit="min"
        )
        forecast_chart = px.line(
            forecast.assign(Forecast_time=forecast_times),
            x="Forecast_time",
            y="predicted_power_kw",
            markers=True,
            labels={
                "Forecast_time": "Time",
                "predicted_power_kw": "Predicted power (kW)",
            },
        )
        forecast_chart.update_layout(showlegend=False)
        st.plotly_chart(forecast_chart, use_container_width=True)

    with chart_right:
        st.subheader("Daily consumption trend")
        daily_energy = (series.resample("D").sum() * 0.25).rename("Energy (kWh)")
        daily_frame = daily_energy.tail(30).rename_axis("Date").reset_index()
        daily_chart = px.bar(
            daily_frame,
            x="Date",
            y="Energy (kWh)",
            labels={"Date": "Date"},
        )
        st.plotly_chart(daily_chart, use_container_width=True)

    st.subheader("Current vs. predicted consumption")
    comparison = go.Figure(
        data=[
            go.Bar(
                x=["Current", f"Predicted (+{horizon_minutes} min)"],
                y=[current_power_kw, predicted_power_kw],
                marker_color=["#2a9d8f", "#e76f51"],
                text=[f"{current_power_kw:.2f}", f"{predicted_power_kw:.2f}"],
                textposition="auto",
            )
        ]
    )
    comparison.update_layout(
        yaxis_title="Power (kW)",
        showlegend=False,
        margin={"t": 20},
    )
    st.plotly_chart(comparison, use_container_width=True)


if __name__ == "__main__":
    main()
