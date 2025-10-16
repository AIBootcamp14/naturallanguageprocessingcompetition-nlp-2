# src/inference/inference_llm.py
import os
import re
import torch
from typing import Optional, List
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
from src.data.dataset_llm import create_inference_prompt


def _clean_generated_summary(text: str) -> str:
    """사용자 요약 출력에서 대화 라인 등을 제거해 한 문단으로 정리합니다."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # 화자 태그로 시작하는 라인 제거 (#PersonX#: ...)
    filtered = [line for line in lines if not re.match(
        r'^#Person\d+#\s*[:：]', line)]

    if not filtered:
        filtered = [re.sub(r'^#Person\d+#\s*[:：]\s*', '', line)
                    for line in lines]

    summary = ' '.join(filtered)
    summary = re.sub(r'#Person(\d+)#', r'#Person\1#', summary)
    summary = re.sub(r'\s+', ' ', summary)
    return summary.strip()


def _needs_fallback(summary: str) -> bool:
    if not summary:
        return True
    if summary.count('?') >= 1:
        return True
    if re.search(r'(\b\w+\b)(\s+\1){2,}', summary):
        return True
    # detect repeated "#PersonX#가 #PersonY#에게" style phrases
    if re.search(r'(#[Pp]erson\d+#\s*\S+\s+#[Pp]erson\d+#\S*){3,}', summary):
        return True
    return False


def _fallback_summary_from_dialogue(dialogue: str) -> str:
    utterances: List[str] = []
    for raw_line in dialogue.splitlines():
        m = re.match(r'^#Person\d+#\s*[:：]\s*(.*)$', raw_line.strip())
        if m:
            utterances.append(m.group(1).strip())

    if not utterances:
        return dialogue.strip()[:200]

    if len(utterances) == 1:
        return utterances[0]

    first = utterances[0]
    last = utterances[-1]
    if first and last and first != last:
        return f"{first} {last}"
    return first


def should_use_few_shot_for_dialogue(dialogue_text: str, few_shot_disable_chars: Optional[int]) -> bool:
    """대화 길이를 기준으로 few-shot 사용 여부를 결정합니다."""
    if few_shot_disable_chars is None:
        return True
    return len(dialogue_text) <= few_shot_disable_chars


def run_inference(
    base_model_id: str,
    lora_adapter_path: str,
    test_data_path: str,
    output_path: str,
    batch_size: int = 4,
    few_shot_examples_path: Optional[str] = None,
    num_fewshot: int = 2,
    few_shot_max_chars: Optional[int] = None,
    few_shot_disable_chars: Optional[int] = 1100,
    use_cpu: bool = False,
):
    """
    파인튜닝된 LLM(QLoRA)을 사용하여 추론을 실행하고 제출 파일을 생성합니다.
    """
    load_dotenv()

    # GPU 사용 가능 여부 확인
    if not use_cpu and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU inference...")
        use_cpu = True

    device = "cpu" if use_cpu else "cuda"
    print(f"Using device: {device}")

    # --- 1. 모델 및 토크나이저 로드 ---
    print(f"Loading base model: {base_model_id}")

    if use_cpu:
        # CPU 추론을 위한 설정
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            token=os.getenv("HF_TOKEN")
        )
    else:
        # GPU 추론을 위한 4-bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- 2. LoRA 어댑터 병합 ---
    print(f"Loading and merging LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    model_device = next(model.parameters()).device

    # --- 3. 테스트 데이터 및 프롬프트 준비 ---
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    few_shot_samples = None
    if few_shot_examples_path:
        if os.path.exists(few_shot_examples_path):
            try:
                source_df = pd.read_csv(few_shot_examples_path)
                if not source_df.empty and few_shot_max_chars is not None:
                    filtered_df = source_df[source_df['dialogue'].str.len()
                                            <= few_shot_max_chars]
                    if len(filtered_df) >= num_fewshot:
                        source_df = filtered_df
                    else:
                        print(
                            f"  Warning: only {len(filtered_df)} few-shot candidates <= {few_shot_max_chars} chars; using full dataset for inference prompts.")
                if not source_df.empty and num_fewshot > 0:
                    few_shot_samples = source_df.sample(
                        n=min(num_fewshot, len(source_df)), random_state=42)
            except Exception as exc:
                print(
                    f"  Failed to load few-shot examples ({exc}). Proceeding without few-shot prompts.")
        else:
            print(
                f"  Few-shot examples path '{few_shot_examples_path}' not found. Proceeding without few-shot prompts.")

    prompts = []
    for _, row in test_df.iterrows():
        use_few_shot = should_use_few_shot_for_dialogue(
            row['dialogue'], few_shot_disable_chars)
        prompt = create_inference_prompt(
            row,
            few_shot_samples=few_shot_samples if use_few_shot else None
        )
        prompts.append(prompt)

    # --- 4. 배치 단위로 요약 생성 ---
    results = []
    print("Generating summaries...")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        batch_rows = test_df.iloc[i:i+batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.no_grad():
            # CPU/GPU에 따른 생성 파라미터 최적화
            generation_kwargs = {
                "max_new_tokens": 500,
                "num_beams": 3,
                "early_stopping": True,
                "repetition_penalty": 1.2,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # CPU 추론 시 더 빠른 설정
            if use_cpu:
                generation_kwargs.update({
                    "num_beams": 1,  # Greedy decoding for speed
                    "do_sample": False,
                })

            outputs = model.generate(**inputs, **generation_kwargs)

        sequence_length = inputs['input_ids'].shape[1]
        for idx, output in enumerate(outputs):
            generated_summary = tokenizer.decode(
                output[sequence_length:], skip_special_tokens=True)
            cleaned = _clean_generated_summary(generated_summary)
            if _needs_fallback(cleaned):
                cleaned = _fallback_summary_from_dialogue(
                    batch_rows.iloc[idx]['dialogue'])
            results.append(cleaned)

    # --- 5. 제출 파일 생성 ---
    print(f"Saving submission file to: {output_path}")
    submission_df = pd.DataFrame({
        'fname': test_df['fname'],
        'summary': results
    })

    # Sample submission 형식에 맞게 인덱스 컬럼 추가
    submission_df.reset_index(inplace=True)
    submission_df.rename(columns={'index': ''}, inplace=True)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submission_df.to_csv(output_path, index=False)
    print("Inference complete!")
