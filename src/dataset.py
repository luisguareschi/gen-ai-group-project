"""Dataset utilities for the Kaggle Fake-and-Real news dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR

KAGGLE_HANDLE = "clmentbisaillon/fake-and-real-news-dataset"


def download_dataset(target_dir: Path | None = None) -> Path:
    """Download the Kaggle fake/real news dataset via kagglehub.

    Returns the path to the directory containing ``Fake.csv`` and ``True.csv``.
    """
    target_dir = target_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    return path


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    """Load Fake.csv + True.csv into a single labelled DataFrame.

    Columns: ``title``, ``text``, ``label`` (0 = REAL, 1 = FAKE), ``article_id``.
    """
    if data_path is None:
        data_path = download_dataset()

    data_path = Path(data_path)
    fake_path = data_path / "Fake.csv"
    real_path = data_path / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            f"Expected Fake.csv and True.csv in {data_path}. "
            "Run scripts/download_dataset.py first."
        )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = 1  # FAKE
    real["label"] = 0  # REAL

    df = pd.concat([fake, real], ignore_index=True)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["article_id"] = df.index.astype(str)
    keep = ["article_id", "title", "text", "label"]
    return df[keep]


def sample_dataset(n: int = 100, random_state: int = 42) -> pd.DataFrame:
    """Return a balanced random sample of ``n`` articles."""
    df = load_dataset()
    half = n // 2
    fake = df[df["label"] == 1].sample(n=half, random_state=random_state)
    real = df[df["label"] == 0].sample(n=n - half, random_state=random_state)
    return (
        pd.concat([fake, real])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def label_to_str(label: int) -> str:
    return "FAKE" if int(label) == 1 else "REAL"
