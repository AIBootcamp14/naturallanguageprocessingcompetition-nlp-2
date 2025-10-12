# src/data/merge_data.py
import os
from typing import Iterable, Optional

import pandas as pd


def merge_datasets(
    base_path: Optional[str],
    augmented_paths: Iterable[str],
    output_path: str,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Merge base dataset with augmented datasets and persist to output_path."""
    frames: list[pd.DataFrame] = []

    if base_path and os.path.exists(base_path):
        frames.append(pd.read_csv(base_path))
    else:
        if base_path:
            print(
                f"Base dataset {base_path} not found. Proceeding without base data.")

    for path in augmented_paths:
        if not path:
            continue
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        else:
            print(f"Augmented dataset {path} not found. Skipping.")

    if not frames:
        raise ValueError("No datasets available to merge.")

    merged_df = pd.concat(frames, ignore_index=True)
    if shuffle and not merged_df.empty:
        merged_df = merged_df.sample(
            frac=1, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path} (size: {len(merged_df)})")

    return merged_df
