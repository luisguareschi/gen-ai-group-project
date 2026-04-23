"""Batch-process articles from the algozee/fake-news Kaggle dataset.

Usage:
    python scripts/run_batch_algozee.py --n 100 --out results/results_algozee.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import kagglehub

from src.config import RESULTS_DIR
from src.crew import run_pipeline

KAGGLE_HANDLE = "algozee/fake-news"

CSV_FIELDS = [
    "article_id",
    "title",
    "true_label",
    "label",
    "confidence",
    "summary",
    "tone",
    "bias_score",
    "claims",
    "fact_check",
    "bias",
    "elapsed_s",
    "error",
]


def load_dataset(n: int, random_state: int) -> pd.DataFrame:
    path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    print(f"[Dataset] Downloaded to {path}")

    csv_files = list(path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    frames = []
    for f in csv_files:
        print(f"[Dataset] Loading {f.name}")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"[Dataset] Columns: {list(df.columns)}")

    df.columns = df.columns.str.lower().str.strip()

    text_col = next((c for c in ["text", "full_text", "body", "content", "article"] if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"Cannot find text column. Available: {list(df.columns)}")

    title_col = next((c for c in ["title", "headline", "subject"] if c in df.columns), "")

    label_col = next((c for c in ["label", "class", "fake", "is_fake", "target"] if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"Cannot find label column. Available: {list(df.columns)}")

    df = df.dropna(subset=[text_col])
    if "publish_date" in df.columns:
        df = df[pd.to_datetime(df["publish_date"], errors="coerce") >= "2024-07-01"]
        print(f"[Dataset] After mid-2024 filter: {len(df)} articles")
    df = df.reset_index(drop=True)

    df["art_text"] = df[text_col].astype(str)
    df["art_title"] = df[title_col].astype(str) if title_col else ""
    df["raw_label"] = df[label_col]

    sample_val = str(df["raw_label"].dropna().iloc[0]).strip().upper()
    if sample_val in ("0", "1"):
        df["art_label"] = df["raw_label"].astype(int)
    else:
        df["art_label"] = df["raw_label"].str.upper().str.strip().map(
            {"FAKE": 1, "FALSE": 1, "1": 1, "REAL": 0, "TRUE": 0, "0": 0}
        ).fillna(0).astype(int)

    df["article_id"] = df.index.astype(str)

    half = n // 2
    fake = df[df["art_label"] == 1].sample(n=min(half, (df["art_label"] == 1).sum()), random_state=random_state)
    real = df[df["art_label"] == 0].sample(n=min(n - half, (df["art_label"] == 0).sum()), random_state=random_state)
    sample = pd.concat([fake, real]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[Dataset] Sampled {len(sample)} articles ({len(fake)} FAKE, {len(real)} REAL)")
    return sample


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["article_id"])["article_id"].astype(str))
    except Exception:
        return set()


def ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()


def append_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "results_algozee.csv")
    args = parser.parse_args()

    sample = load_dataset(args.n, args.seed)
    processed = load_processed_ids(args.out)
    ensure_header(args.out)

    remaining = sample[~sample["article_id"].astype(str).isin(processed)]
    print(f"{len(processed)} already done; {len(remaining)} to process. Writing to {args.out}")

    for i, rec in enumerate(remaining.itertuples(index=False), 1):
        article_id = str(rec.article_id)
        title = str(rec.art_title or "")
        text = str(rec.art_text or "")
        true_label = "FAKE" if int(rec.art_label) == 1 else "REAL"

        print(f"\n=== [{i}/{len(remaining)}] article_id={article_id} (true={true_label}) ===")
        start = time.time()
        row: dict = {"article_id": article_id, "title": title[:200], "true_label": true_label}
        try:
            result = run_pipeline(title, text)
            flat = result.to_row(article_id)
            row.update({k: flat.get(k, "") for k in CSV_FIELDS if k in flat})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        row["elapsed_s"] = round(time.time() - start, 2)
        append_row(args.out, row)
        print(f"-> label={row.get('label')!r} conf={row.get('confidence')!r} ({row['elapsed_s']}s)")


if __name__ == "__main__":
    main()
