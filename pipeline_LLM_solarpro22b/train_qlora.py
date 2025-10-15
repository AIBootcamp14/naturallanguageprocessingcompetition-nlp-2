"""QLoRA fine-tuning for Solar-Pro on dialogue summarization (trl 0.23.1)"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

# ==================== 설정 ====================

MODEL_ID = "upstage/solar-pro-preview-instruct"
OUTPUT_DIR = "output/solar-pro-qlora-summarization"

# QLoRA 설정
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 학습 하이퍼파라미터
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
SAVE_STEPS = 50
MAX_SEQ_LENGTH = 2048

# ==================== 모델 로드 ====================


def load_model_and_tokenizer():
    """4bit 양자화로 모델 로드"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # Fine-tuning 시 right padding

    return model, tokenizer

# ==================== LoRA 설정 ====================


def setup_lora(model):
    """LoRA 어댑터 추가"""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ==================== 학습 ====================


def main():
    print("="*60)
    print("QLoRA Fine-tuning: Solar-Pro for Dialogue Summarization")
    print("="*60)

    # 1. 데이터 로드
    print("\n[1/5] Loading datasets...")
    train_dataset = load_from_disk("data/train_formatted")
    eval_dataset = load_from_disk("data/dev_formatted")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval: {len(eval_dataset)} samples")

    # 2. 모델 & 토크나이저
    print("\n[2/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # 3. LoRA 설정
    print("\n[3/5] Setting up LoRA...")
    model = setup_lora(model)

    # 4. 포매팅 함수
    print("\n[4/5] Preparing formatter...")

    def formatting_func(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text

    # 5. 데이터 전처리
    print("\n[5/5] Preprocessing data...")

    def preprocess(examples):
        texts = []
        for msg in examples["messages"]:
            text = formatting_func({"messages": msg})
            texts.append(text)

        # 토큰화
        tokenized = tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,  # Trainer가 batch마다 padding
        )

        return tokenized

    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # 6. 학습 설정
    print("\n[6/6] Starting training...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # 학습 파라미터
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,

        # 스케줄러
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,

        # 로깅 & 체크포인트
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        eval_strategy="steps",
        save_total_limit=3,

        # 최적화
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",

        # 기타
        report_to="none",
    )

    # Trainer 생성 (max_seq_length는 여기에)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 학습 시작
    trainer.train()

    # 저장
    print("\n✓ Training complete!")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
