"""Batch-process a sample of articles through the multi-agent pipeline.

Writes each result incrementally to the output CSV for resumability.

Usage:
    python scripts/run_batch.py --n 100 --out results/results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.crew import run_pipeline  # noqa: E402
from src.dataset import label_to_str, sample_dataset  # noqa: E402


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


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["article_id"])
        return set(df["article_id"].astype(str))
    except Exception:
        return set()


def ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of articles to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    sample = sample_dataset(n=args.n, random_state=args.seed)
    processed = load_processed_ids(args.out)
    ensure_header(args.out)

    remaining = sample[~sample["article_id"].astype(str).isin(processed)]
    print(f"Sampled {len(sample)} articles; {len(processed)} already done; "
          f"{len(remaining)} to process. Writing to {args.out}")

    for i, rec in enumerate(remaining.itertuples(index=False), 1):
        article_id = str(rec.article_id)
        title = str(rec.title or "")
        text = str(rec.text or "")
        true_label = label_to_str(rec.label)

        print(f"\n=== [{i}/{len(remaining)}] article_id={article_id} (true={true_label}) ===")
        start = time.time()
        row: dict = {
            "article_id": article_id,
            "title": title[:200],
            "true_label": true_label,
        }
        try:
            result = run_pipeline(title, text)
            flat = result.to_row(article_id)
            row.update({k: flat.get(k, "") for k in CSV_FIELDS if k in flat})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        row["elapsed_s"] = round(time.time() - start, 2)
        append_row(args.out, row)
        print(f"-> label={row.get('label')!r} conf={row.get('confidence')!r} "
              f"({row['elapsed_s']}s)")


if __name__ == "__main__":
    main()
