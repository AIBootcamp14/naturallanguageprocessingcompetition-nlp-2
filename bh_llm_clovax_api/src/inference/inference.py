"""Pipeline to fill test.csv using HyperCLOVA X without local training."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.local_clovax import LocalClovaX
from src.inference.prompt_builder import build_messages
from src.utils.config import load_config
from src.utils.examples import examples_to_records, select_fewshot_examples
from src.utils.io import export_jsonl, load_csv, write_csv
import os
import warnings
import logging as _logging
import pandas as pd
from math import ceil

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyperCLOVA X inference on test.csv")
    parser.add_argument("--cfg", default="configs/clovax.yaml",
                        help="Path to YAML config")
    return parser.parse_args()


def _parse_generation_content(content: str | None) -> str:
    """
    모델 응답에서 summary 문자열만 꺼내서 반환.
    - JSON {"summary":"..."} 우선
    - 이후 정규식으로 "summary": "..." 만 추출
    """
    import json
    import re
    if not content:
        return ""
    text = content.strip()
    if not text:
        return ""

    # 1) JSON 그대로 시도
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data:
            return str(data["summary"]).strip()
    except json.JSONDecodeError:
        pass

    # 2) 본문에서 {...} 블록 찾아 JSON 재시도
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            data = json.loads(text[first:last+1])
            if isinstance(data, dict) and "summary" in data:
                return str(data["summary"]).strip()
        except json.JSONDecodeError:
            pass

    # 3) "summary": " ... " 패턴에서 큰따옴표 내부만
    m = re.search(r'summary\s*:\s*"(.*?)"', text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'summary\s*"\s*:\s*"(.*?)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 4) 최후: 첫 줄만
    return text.splitlines()[0].strip()


def run_pipeline(cfg_or_path) -> dict:
    """
    - cfg_or_path: dict 또는 YAML 경로(str/Path)
    - sample_submission.csv의 스키마(열/순서) 그대로 유지
    - test.csv의 fname 매칭으로 summary만 채움
    - 프로그레스바는 '대화 수' 기준으로 1개만 표시
    """

    # ---- 0) 설정 로드 (경로/사전 모두 허용) ----
    if isinstance(cfg_or_path, (str, Path)):
        cfg = load_config(str(cfg_or_path))
    elif isinstance(cfg_or_path, dict):
        cfg = cfg_or_path
    else:
        raise TypeError(f"Unsupported cfg type: {type(cfg_or_path)}")

    # ---- 1) 경로/입력 로드 ----
    data_root = Path(cfg["data"]["root"])
    test_path = data_root / cfg["data"]["test_file"]
    sample_path = data_root / cfg["data"]["sample_submission"]

    test_df = load_csv(str(test_path))
    submission_template = load_csv(str(sample_path))

    # few-shot 예제 (train.csv는 few-shot 선택용으로만 사용)
    train_path = data_root / cfg["data"]["train_file"]
    try:
        examples = select_fewshot_examples(
            train_path=str(train_path),
            sample_count=int(cfg["prompt"]["fewshot"]["sample_count"]),
            topic_round_robin=bool(
                cfg["prompt"]["fewshot"]["topic_round_robin"]),
            seed=int(cfg["prompt"]["fewshot"]["seed"]),
            export_meta=bool(cfg["prompt"]["fewshot"]["export_meta"]),
            example_template=cfg["prompt"]["fewshot"]["example_template"],
            out_path=cfg["output"]["fewshot_export"],
        )
    except Exception:
        examples = []

    # ---- 2) 모델/토크나이저 준비 ----
    model_cfg = cfg["model"]
    generation_cfg = cfg["generation"]
    client = LocalClovaX(model_cfg, generation_cfg)

    # ---- 3) 진행 로그/경고 정리 (tqdm 깨짐 방지) ----
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore", message="Starting from v4.46, the `logits` model output")
    _logging.getLogger("accelerate").setLevel(_logging.WARNING)
    _logging.getLogger(
        "transformers.generation.configuration_utils").setLevel(_logging.ERROR)

    # ---- 4) 배치 추론 (tqdm 1개) ----
    rows = list(test_df.itertuples(index=False))
    total = len(rows)
    batch_size = int(cfg.get("inference", {}).get("batch_size", 16))
    use_bar = bool(cfg.get("logging", {}).get("use_tqdm", False))

    def _iter_batches(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    def _parse_generation_content(content: str | None) -> str:
        import json
        import re
        if not content:
            return ""
        text = content.strip()
        if not text:
            return ""
        # 1) 전체 JSON 시도
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "summary" in data:
                return str(data["summary"]).strip()
        except json.JSONDecodeError:
            pass
        # 2) {...} 블록 추출 후 재시도
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                data = json.loads(text[first:last+1])
                if isinstance(data, dict) and "summary" in data:
                    return str(data["summary"]).strip()
            except json.JSONDecodeError:
                pass
        # 3) "summary": " ... "
        m = re.search(r'summary\s*:\s*"(.*?)"', text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r'summary\s*"\s*:\s*"(.*?)"',
                          text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # 4) 최후: 첫 줄
        return text.splitlines()[0].strip()

    detailed_rows = []
    pbar = None
    batches = _iter_batches(rows, batch_size)
    if use_bar:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="ClovaX inference",
                    unit="dialogue", dynamic_ncols=True)

    for batch in batches:
        # test.csv에서 대화 컬럼명: 'dialogue' 우선, 없으면 'text'/'content'
        messages_list = []
        for r in batch:
            dialogue = getattr(r, "dialogue", None)
            if dialogue is None:
                dialogue = getattr(r, "text", None)
            if dialogue is None:
                dialogue = getattr(r, "content", "")
            messages_list.append(build_messages(
                dialogue, cfg["prompt"], examples))

        # 배치 생성 (생성 텍스트만 반환)
        texts = client.generate_batch(messages_list)

        # summary만 추출/적재
        for r, txt in zip(batch, texts):
            detailed_rows.append({
                "fname": getattr(r, "fname"),
                "summary": _parse_generation_content(txt),
                "raw_content": txt,
            })

        if pbar:
            pbar.update(len(batch))
    if pbar:
        pbar.close()

    # ---- 5) 저장: 상세/제출 (제출은 sample 스키마 그대로) ----
    detailed_df = pd.DataFrame(detailed_rows).reindex(
        columns=["fname", "summary", "raw_content"])
    detailed_path = Path(cfg["output"]["detailed_csv"])
    write_csv(detailed_df, str(detailed_path), index=False)

    submission_df = submission_template.copy()
    if "fname" not in submission_df.columns:
        raise KeyError("sample_submission.csv must contain a 'fname' column")

    summary_map = {row["fname"]: row["summary"] for row in detailed_rows}
    if "summary" not in submission_df.columns:
        submission_df["summary"] = ""
    submission_df["summary"] = submission_df["fname"].map(
        summary_map).fillna("")
    submission_df = submission_df.reindex(
        columns=list(submission_template.columns))

    submission_path = Path(cfg["output"]["submission_csv"])
    write_csv(submission_df, str(submission_path), index=False)

    return {
        "num_items": total,
        "detailed_csv": str(detailed_path),
        "submission_csv": str(submission_path),
        "submission_path": str(submission_path),  # main()의 logger 호환용
    }


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.cfg)
    logger.info("Saved submission to %s", result["submission_path"])


if __name__ == "__main__":
    main()
