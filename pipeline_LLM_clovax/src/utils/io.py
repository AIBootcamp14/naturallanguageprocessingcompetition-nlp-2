"""I/O helpers for the NLP ClovaX pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


__all__ = [
    "ensure_dir",
    "load_csv",
    "write_csv",
    "export_jsonl",
]


def ensure_dir(path: Path | str) -> Path:
    dst = Path(path)
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def load_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def write_csv(df: pd.DataFrame, path: Path | str, index: bool = False) -> Path:
    dst = Path(path)
    ensure_dir(dst.parent)
    df.to_csv(dst, index=index)
    return dst


def export_jsonl(records: Iterable[dict], path: Path | str) -> Path:
    dst = Path(path)
    ensure_dir(dst.parent)
    with dst.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dst
