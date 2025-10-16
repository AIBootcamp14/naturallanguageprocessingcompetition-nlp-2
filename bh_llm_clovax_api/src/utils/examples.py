"""Few-shot selection utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class FewShotExample:
    fname: str
    dialogue: str
    summary: str
    topic: str

    @classmethod
    def from_series(cls, row: pd.Series) -> "FewShotExample":
        return cls(
            fname=str(row["fname"]),
            dialogue=str(row["dialogue"]),
            summary=str(row.get("summary", "")),
            topic=str(row.get("topic", "")),
        )


def select_fewshot_examples(
    train_df: pd.DataFrame,
    sample_count: int,
    seed: int = 42,
    topic_round_robin: bool = True,
) -> list[FewShotExample]:
    """Select representative few-shot examples from train.csv."""
    if sample_count <= 0:
        return []
    sample_count = min(sample_count, len(train_df))
    rng = random.Random(seed)

    examples: list[FewShotExample] = []
    if topic_round_robin and "topic" in train_df.columns:
        grouped = {
            topic: grp.sample(frac=1, random_state=seed).reset_index(drop=True)
            for topic, grp in train_df.groupby("topic")
        }
        topics = list(grouped.keys())
        rng.shuffle(topics)
        topic_index = {topic: 0 for topic in topics}

        while len(examples) < sample_count and topics:
            for topic in list(topics):
                grp = grouped[topic]
                idx = topic_index[topic]
                if idx >= len(grp):
                    topics.remove(topic)
                    continue
                row = grp.iloc[idx]
                topic_index[topic] += 1
                examples.append(FewShotExample.from_series(row))
                if len(examples) >= sample_count:
                    break
    else:
        sampled = train_df.sample(n=sample_count, random_state=seed)
        examples = [FewShotExample.from_series(row) for _, row in sampled.iterrows()]

    return examples[:sample_count]


def examples_to_records(items: Iterable[FewShotExample]) -> list[dict]:
    return [
        {
            "fname": ex.fname,
            "summary": ex.summary,
            "topic": ex.topic,
            "dialogue": ex.dialogue,
        }
        for ex in items
    ]
