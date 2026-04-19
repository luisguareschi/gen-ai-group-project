"""Download the Kaggle Fake-and-Real news dataset via kagglehub."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import download_dataset, load_dataset  # noqa: E402


def main() -> None:
    path = download_dataset()
    print(f"Dataset downloaded to: {path}")
    df = load_dataset(path)
    print(f"Loaded {len(df)} articles total.")
    print(df["label"].value_counts().rename({0: "REAL", 1: "FAKE"}))


if __name__ == "__main__":
    main()
