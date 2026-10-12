"""Train, evaluate, and save the household energy LSTM model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import SEQUENCE_LENGTH, load_and_preprocess_data
from src.utils import MODELS_DIR, ensure_project_directories

MODEL_PATH = MODELS_DIR / "lstm_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
LOSS_PLOT_PATH = MODELS_DIR / "training_loss.png"


def build_model(sequence_length: int = SEQUENCE_LENGTH) -> tf.keras.Model:
    """Construct the project's two-layer LSTM regression model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _save_loss_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation loss curves."""
    training_loss = history.history.get("loss")
    validation_loss = history.history.get("val_loss")
    if not training_loss or not validation_loss:
        raise RuntimeError(
            "Training did not produce both training and validation loss."
        )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(training_loss, label="Training loss")
    axis.plot(validation_loss, label="Validation loss")
    axis.set(title="LSTM Training and Validation Loss", xlabel="Epoch", ylabel="MSE")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train_model(
    epochs: int = 20,
    batch_size: int = 32,
) -> dict[str, float | int]:
    """Train the LSTM and save its model, scaler, evaluation, and loss plot."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    ensure_project_directories()
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    if len(X_train) < 2:
        raise ValueError("At least two training sequences are required for validation.")

    tf.keras.utils.set_random_seed(42)
    model = build_model(input_sequence_length(X_train))
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=False,
        verbose=1,
    )

    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).ravel()
    actual = scaler.inverse_transform(y_test).ravel()
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
        "test_samples": int(len(actual)),
        "sequence_length": int(X_train.shape[1]),
    }

    model.save(MODEL_PATH)
    dump(scaler, SCALER_PATH)
    _save_loss_plot(history, LOSS_PLOT_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")
    print("Holdout metrics (kW):")
    for name in ("mae", "rmse", "r2"):
        print(f"  {name.upper()}: {metrics[name]:.4f}")
    return metrics


def input_sequence_length(X_train: np.ndarray) -> int:
    """Return the sequence length, validating the expected model input shape."""
    if X_train.ndim != 3 or X_train.shape[2] != 1:
        raise ValueError(
            "Expected training data shaped as (samples, timesteps, 1 feature)."
        )
    return int(X_train.shape[1])


def main() -> None:
    """Run training using optional environment-based configuration."""
    epochs = int(os.environ.get("EPOCHS", "20"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    train_model(epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    main()
