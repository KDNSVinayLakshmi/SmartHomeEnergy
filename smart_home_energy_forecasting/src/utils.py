"""Shared project paths and UCI dataset ingestion helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "household_power_consumption.csv"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


def ensure_project_directories() -> None:
    """Create runtime directories for downloaded data and trained artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(destination: Path = DATASET_PATH) -> Path:
    """Download and convert the official UCI archive to a CSV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(DATASET_URL, timeout=60) as response:
            archive_bytes = response.read()
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download the UCI household power dataset from "
            f"{DATASET_URL}: {exc}"
        ) from exc

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            member_name = next(
                (
                    name
                    for name in archive.namelist()
                    if name.lower().endswith("household_power_consumption.txt")
                ),
                None,
            )
            if member_name is None:
                raise RuntimeError(
                    "The UCI archive did not contain "
                    "household_power_consumption.txt."
                )

            with archive.open(member_name) as source:
                frame = pd.read_csv(
                    source,
                    sep=";",
                    na_values=["?"],
                    low_memory=False,
                )
    except BadZipFile as exc:
        raise RuntimeError(
            "The downloaded UCI dataset is not a valid ZIP archive."
        ) from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read the UCI dataset archive: {exc}") from exc

    required_columns = {"Date", "Time", "Global_active_power"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "The UCI dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame.to_csv(destination, index=False)
    return destination
