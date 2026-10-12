"""Load saved forecasting artifacts and recursively predict up to one hour."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from sklearn.preprocessing import MinMaxScaler

from src.data_preprocessing import SEQUENCE_LENGTH
from src.utils import MODELS_DIR

MODEL_PATH = MODELS_DIR / "lstm_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FORECAST_STEPS = 4


def load_model_and_scaler(
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
) -> tuple[tf.keras.Model, MinMaxScaler]:
    """Load the trained Keras model and fitted scaler from disk."""
    model_file = Path(model_path)
    scaler_file = Path(scaler_path)
    missing = [str(path) for path in (model_file, scaler_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Forecast artifacts are missing: "
            + ", ".join(missing)
            + ". Train the model first with `python -m src.train_model`."
        )

    try:
        model = tf.keras.models.load_model(model_file)
        scaler = load(scaler_file)
    except (OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not load forecast artifacts: {exc}") from exc
    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(f"Expected a fitted MinMaxScaler in {scaler_file}.")
    return model, scaler


def forecast_next_hour(
    latest_readings: Sequence[float] | pd.Series | np.ndarray,
    model: tf.keras.Model | None = None,
    scaler: MinMaxScaler | None = None,
) -> pd.DataFrame:
    """Return recursive predictions at 15-minute steps for the next hour."""
    values = np.asarray(latest_readings, dtype=np.float64).reshape(-1, 1)
    if len(values) < SEQUENCE_LENGTH:
        raise ValueError(
            f"At least {SEQUENCE_LENGTH} recent readings are required; "
            f"received {len(values)}."
        )
    if not np.isfinite(values[-SEQUENCE_LENGTH:]).all():
        raise ValueError("The latest 96 readings must all be finite numbers.")

    if model is None or scaler is None:
        loaded_model, loaded_scaler = load_model_and_scaler()
        if model is None:
            model = loaded_model
        if scaler is None:
            scaler = loaded_scaler

    window = scaler.transform(values[-SEQUENCE_LENGTH:]).reshape(
        1, SEQUENCE_LENGTH, 1
    )
    predictions: list[float] = []
    for _ in range(FORECAST_STEPS):
        prediction_scaled = np.asarray(
            model.predict(window, verbose=0)
        ).reshape(-1, 1)
        if (
            prediction_scaled.shape != (1, 1)
            or not np.isfinite(prediction_scaled).all()
        ):
            raise RuntimeError("The model returned an invalid one-step forecast.")
        prediction = float(scaler.inverse_transform(prediction_scaled)[0, 0])
        predictions.append(max(prediction, 0.0))
        window = np.concatenate(
            (window[:, 1:, :], prediction_scaled.reshape(1, 1, 1)),
            axis=1,
        )

    return pd.DataFrame(
        {
            "minutes_ahead": [15, 30, 45, 60],
            "predicted_power_kw": predictions,
        }
    )
