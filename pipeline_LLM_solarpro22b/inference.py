"""
Test 데이터 추론 - Fine-tuned Solar Pro로 대화 요약 생성 (배치 처리)
"""
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ==================== 설정 ====================

MODEL_DIR = "output/solar-pro-qlora-summarization"
BASE_MODEL_ID = "upstage/solar-pro-preview-instruct"

TEST_FILE = "data/test.csv"
OUTPUT_FILE = "sample_submission.csv"

MAX_NEW_TOKENS = 512
BATCH_SIZE = 2

# ==================== 모델 로드 ====================


def load_finetuned_model():
    """Fine-tuned 모델 로드"""
    print("Loading base model...")

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    base_model.config.use_cache = False

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"  # 배치 처리용

    print("✓ Model loaded successfully\n")
    return model, tokenizer


# ==================== 추론 ====================

def generate_summaries_batch(dialogues: list, model, tokenizer) -> list:
    """여러 대화문을 배치로 요약 생성"""

    all_prompts = []
    for dialogue in dialogues:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes dialogues concisely."
            },
            {
                "role": "user",
                "content": f"Summarize the following dialogue:\n\n{dialogue}"
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_prompts.append(prompt)

    inputs = tokenizer(
        all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )

    summaries = []
    for i, output in enumerate(outputs):
        prompt_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum()
        summary = tokenizer.decode(
            output[prompt_len:],
            skip_special_tokens=True
        ).strip()
        summaries.append(summary)

    del inputs, outputs
    torch.cuda.empty_cache()

    return summaries


# ==================== 메인 ====================

def main():
    print("="*70)
    print("Test Data Inference - Solar Pro Fine-tuned Model")
    print("="*70)

    model, tokenizer = load_finetuned_model()

    print(f"Loading test data from {TEST_FILE}...")
    test_df = pd.read_csv(TEST_FILE)
    print(f"✓ Loaded {len(test_df)} test samples\n")

    print("="*70)
    print(f"Generating summaries (Batch size: {BATCH_SIZE})...")
    print("="*70)

    all_summaries = []

    for batch_start in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(test_df))
        batch_df = test_df.iloc[batch_start:batch_end]

        dialogues = batch_df['dialogue'].tolist()

        try:
            summaries = generate_summaries_batch(dialogues, model, tokenizer)
            all_summaries.extend(summaries)

            if (batch_end) % 20 == 0:
                print(f"\n[Progress] {batch_end}/{len(test_df)} samples")

        except Exception as e:
            print(f"\n[ERROR] Batch {batch_start}-{batch_end}: {e}")
            all_summaries.extend([""] * len(dialogues))

    print(f"\n{'='*70}")
    print("Saving results...")

    submission_df = pd.DataFrame({
        'fname': test_df['fname'],
        'summary': all_summaries
    })

    submission_df.insert(0, '', range(len(submission_df)))
    submission_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"✓ Saved to {OUTPUT_FILE}")
    print(f"✓ Total samples: {len(submission_df)}")
    print("="*70)

    # 샘플 출력
    print("\n=== Sample Results ===")
    for i in range(min(3, len(submission_df))):
        print(f"\n[Sample {i+1}]")
        print(f"Summary: {submission_df.iloc[i]['summary'][:150]}...")


if __name__ == "__main__":
    main()
