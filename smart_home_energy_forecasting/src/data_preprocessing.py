"""Data loading, cleaning, scaling, and LSTM sequence preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils import DATASET_PATH, download_dataset

SEQUENCE_LENGTH = 96
RESAMPLE_INTERVAL = "15min"
TARGET_COLUMN = "Global_active_power"


def load_consumption_series(
    csv_path: str | Path | None = None,
) -> pd.Series:
    """Load and clean household power readings as 15-minute kW averages."""
    path = Path(csv_path) if csv_path is not None else DATASET_PATH
    if not path.exists():
        path = download_dataset(path)

    try:
        frame = pd.read_csv(path, na_values=["?"])
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Could not read household power data at {path}: {exc}"
        ) from exc

    required_columns = {"Date", "Time", TARGET_COLUMN}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {path} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    timestamps = pd.to_datetime(
        frame["Date"].astype(str).str.strip()
        + " "
        + frame["Time"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    power = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    readings = pd.DataFrame({"timestamp": timestamps, TARGET_COLUMN: power})
    readings = readings.dropna(subset=["timestamp"])
    readings[TARGET_COLUMN] = readings[TARGET_COLUMN].replace(
        [np.inf, -np.inf], np.nan
    )
    readings = readings.dropna(subset=[TARGET_COLUMN])
    if readings.empty:
        raise ValueError(f"Dataset {path} contains no valid power readings.")

    readings = readings.set_index("timestamp").sort_index()
    series = readings[TARGET_COLUMN].resample(RESAMPLE_INTERVAL).mean()
    series = series.interpolate(method="time", limit=4, limit_area="inside")
    series = series.ffill(limit=4)
    series = series.dropna()
    series.name = TARGET_COLUMN

    minimum_readings = SEQUENCE_LENGTH + 2
    if len(series) < minimum_readings:
        raise ValueError(
            f"At least {minimum_readings} valid 15-minute readings are required; "
            f"found {len(series)}."
        )
    return series


def _make_sequences(
    normalized_values: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows and retain each target's original row index."""
    windows = np.lib.stride_tricks.sliding_window_view(
        normalized_values[:, 0], sequence_length + 1
    )
    X = windows[:, :-1, np.newaxis].copy()
    y = windows[:, -1, np.newaxis].copy()
    target_indices = np.arange(sequence_length, len(normalized_values))
    return X, y, target_indices


def load_and_preprocess_data(
    csv_path: str | Path | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare chronological train/test sequences and a training-only scaler."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    series = load_consumption_series(csv_path)
    values = series.to_numpy(dtype=np.float64).reshape(-1, 1)
    split_index = int(len(values) * (1 - test_size))
    if split_index <= sequence_length or split_index >= len(values):
        raise ValueError(
            "The dataset is too short for the selected sequence length and test size."
        )

    scaler = MinMaxScaler()
    scaler.fit(values[:split_index])
    normalized_values = scaler.transform(values)
    X, y, target_indices = _make_sequences(normalized_values, sequence_length)
    train_mask = target_indices < split_index
    test_mask = ~train_mask

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "The dataset did not produce both training and test sequences."
        )

    return (
        X[train_mask].astype(np.float32),
        X[test_mask].astype(np.float32),
        y[train_mask].astype(np.float32),
        y[test_mask].astype(np.float32),
        scaler,
    )
