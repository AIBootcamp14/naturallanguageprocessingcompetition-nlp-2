"""Prompt construction helpers for Clova X summarization."""
from __future__ import annotations

import json
from typing import Sequence

from src.utils.examples import FewShotExample


def _wrap_text(content: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": content.strip()}]


def build_messages(
    dialogue: str,
    prompt_config: dict,
    examples: Sequence[FewShotExample],
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []

    system_prompt = (prompt_config.get("system") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": _wrap_text(system_prompt)})

    user_template = prompt_config.get("user_template", "{dialogue}")
    example_template = (
        prompt_config.get("fewshot", {}).get("example_template")
        or user_template
    )

    for example in examples:
        user_content = example_template.format(
            dialogue=example.dialogue,
            summary=example.summary,
            topic=example.topic,
            fname=example.fname,
        ).strip()
        assistant_content = json.dumps(
            {"summary": example.summary, "topic": example.topic},
            ensure_ascii=False,
        )
        messages.append({"role": "user", "content": _wrap_text(user_content)})
        messages.append({"role": "assistant", "content": _wrap_text(assistant_content)})

    final_user = user_template.format(dialogue=dialogue).strip()
    messages.append({"role": "user", "content": _wrap_text(final_user)})
    return messages
