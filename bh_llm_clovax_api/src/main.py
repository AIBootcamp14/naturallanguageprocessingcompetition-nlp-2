"""Entry point for the NLP ClovaX project."""
from __future__ import annotations

from pathlib import Path

from src.inference.inference import run_pipeline


DEFAULT_CONFIG = Path("configs/clovax.yaml")


def main() -> None:
    result = run_pipeline(DEFAULT_CONFIG)
    print(f"[OK] Submission saved to: {result['submission_path']}")


if __name__ == "__main__":
    main()
