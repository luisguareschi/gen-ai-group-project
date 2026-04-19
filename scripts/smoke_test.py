"""Quick smoke test of the pipeline on a single hand-picked article."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crew import run_pipeline


def main() -> None:
    title = "Scientists confirm the Moon is made of cheese"
    body = (
        "In a shocking announcement today, a team of scientists at Harvard University "
        "declared that the Moon is entirely made of aged cheddar cheese. The study, "
        "published in the Journal of Dairy Astronomy, claims that Apollo 11 astronauts "
        "secretly brought back cheese samples in 1969 that have been hidden from the "
        "public for over five decades."
    )
    result = run_pipeline(title, body)
    print("\n==================== RESULT ====================")
    print("Claims    :", result.claims)
    print("Fact check:", result.fact_check)
    print("Bias      :", result.bias)
    print("Judge     :", result.judge)


if __name__ == "__main__":
    main()
