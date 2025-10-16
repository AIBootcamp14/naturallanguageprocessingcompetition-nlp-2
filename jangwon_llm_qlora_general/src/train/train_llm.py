# src/train/train_llm.py
import os
import random
import torch
import numpy as np
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from evaluate import load
from functools import partial
from src.data.dataset_llm import create_inference_prompt, create_prompt, create_prompt_with_fewshot
from mecab import MeCab


# ROUGE 점수 계산을 위한 전역 metric 객체 로드
rouge_metric = load("rouge")
# 형태소 분석기 초기화
mecab = MeCab()


class PredictionCallback(TrainerCallback):
    """
    매 평가가 끝날 때마다 고정된 dev 샘플 3개에 대한 예측과 정답을 출력하는 콜백.
    """

    def __init__(self, trainer, tokenizer, dev_samples, few_shot_samples=None, disable_fewshot_threshold=None):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dev_samples = dev_samples
        self.few_shot_samples = few_shot_samples
        self.disable_fewshot_threshold = disable_fewshot_threshold

    def on_evaluate(self, args, state, control, **kwargs):
        print("\n\n" + "="*50)
        print("============ EPOCH EVALUATION SAMPLES ============")
        print("="*50)

        accelerator = getattr(self.trainer, "accelerator", None)
        if accelerator is not None:
            model_device = accelerator.device
        else:
            try:
                model_device = next(self.trainer.model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")

        for i, sample in self.dev_samples.iterrows():
            use_dynamic_few_shot = self.few_shot_samples is not None and not getattr(
                self.few_shot_samples, "empty", False)
            if use_dynamic_few_shot and self.disable_fewshot_threshold is not None:
                if len(sample['dialogue']) > self.disable_fewshot_threshold:
                    use_dynamic_few_shot = False

            prompt = create_inference_prompt(
                sample,
                few_shot_samples=self.few_shot_samples if use_dynamic_few_shot else None,
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=3072)
            inputs = {key: value.to(model_device)
                      for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.trainer.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    num_beams=3,
                    early_stopping=True,
                    repetition_penalty=1.2,
                )

            input_length = len(inputs['input_ids'][0])
            generated_summary = self.tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True)

            print(f"\n[Sample {i}]")
            print(f"  PRED: {generated_summary.strip()}")
            print(f"  GOLD: {sample['summary'].strip()}")

        print("="*50)


def _compute_final_metrics(predictions, labels, tokenizer):
    """
    공식 평가 지표에 맞춰 ROUGE 점수 및 Final Score를 계산합니다.
    (형태소 분석 포함)
    """
    predictions = np.where(
        predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True)

    # 예측 결과에서 프롬프트 부분 제거
    cleaned_preds = [pred.split("### 요약:")[-1].strip()
                     for pred in decoded_preds]

    # 형태소 분석 적용
    tokenized_preds = [' '.join(mecab.morphs(pred)) for pred in cleaned_preds]
    tokenized_labels = [' '.join(mecab.morphs(label))
                        for label in decoded_labels]

    # ROUGE 점수 계산
    result = rouge_metric.compute(
        predictions=tokenized_preds, references=tokenized_labels, use_stemmer=True)

    def _extract_rouge(value):
        if hasattr(value, "mid"):
            mid = value.mid
            if hasattr(mid, "fmeasure"):
                return float(mid.fmeasure)
        if isinstance(value, dict):
            if "mid" in value:
                mid = value["mid"]
                if isinstance(mid, dict):
                    return float(mid.get("fmeasure", mid.get("f", 0.0)))
                if hasattr(mid, "fmeasure"):
                    return float(mid.fmeasure)
                return float(mid)
            return float(value.get("fmeasure", value.get("f", 0.0)))
        return float(value)

    # f-measure 점수 추출
    rouge_scores = {key: _extract_rouge(value)
                    for key, value in result.items()}

    # Final Score 계산
    final_score = (rouge_scores.get('rouge1', 0) +
                   rouge_scores.get('rouge2', 0) +
                   rouge_scores.get('rougeL', 0))

    # Wandb 로깅을 위한 최종 결과 포맷팅 (* 100 처리)
    wandb_log = {f"eval_{key}": value *
                 100 for key, value in rouge_scores.items()}
    wandb_log["eval_final_score"] = final_score * 100

    # `metric_for_best_model`이 참조할 수 있도록 final_score 추가
    wandb_log["final_score"] = final_score

    return wandb_log


def build_batch_metrics_fn(tokenizer):
    """
    SFTTrainer에서 batch_eval_metrics=True로 동작할 때 배치 단위 로짓/레이블을
    축적하여 한 번만 ROUGE를 계산하도록 하는 헬퍼.
    """
    preds_buffer = []
    labels_buffer = []

    def _compute_metrics(eval_pred, compute_result=True, **kwargs):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        if preds is not None:
            if isinstance(preds, tuple):
                preds = preds[0]
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            elif not isinstance(preds, np.ndarray):
                preds = np.asarray(preds)
            if preds.ndim == 3:
                preds = preds.argmax(axis=-1)
            preds_buffer.append(preds.astype(np.int32, copy=False))

        if labels is not None:
            if isinstance(labels, tuple):
                labels = labels[0]
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            elif not isinstance(labels, np.ndarray):
                labels = np.asarray(labels)
            labels_buffer.append(labels.astype(np.int32, copy=False))

        if not compute_result:
            return {}

        if not preds_buffer or not labels_buffer:
            preds_buffer.clear()
            labels_buffer.clear()
            return {}

        max_pred_len = max(arr.shape[1] for arr in preds_buffer)
        pad_token = tokenizer.pad_token_id
        if pad_token is None:
            pad_token = getattr(tokenizer, 'eos_token_id', 0)
        preds_padded = [
            arr if arr.shape[1] == max_pred_len
            else np.pad(arr, ((0, 0), (0, max_pred_len - arr.shape[1])), constant_values=pad_token)
            for arr in preds_buffer
        ]

        max_label_len = max(arr.shape[1] for arr in labels_buffer)
        labels_padded = [
            arr if arr.shape[1] == max_label_len
            else np.pad(arr, ((0, 0), (0, max_label_len - arr.shape[1])), constant_values=-100)
            for arr in labels_buffer
        ]

        stacked_preds = np.concatenate(preds_padded, axis=0)
        stacked_labels = np.concatenate(labels_padded, axis=0)
        preds_buffer.clear()
        labels_buffer.clear()
        return _compute_final_metrics(stacked_preds, stacked_labels, tokenizer)

    return _compute_metrics


def train_llm_with_qlora(config: dict):

    # 환경 변수 및 모델 ID 설정
    load_dotenv()
    model_id = config['model']['id']

    # 4비트 양자화를 위한 QLoRA 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 모델 및 토크나이저 로드
    print(f"  Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA 설정
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        # Solar, Qwen2, Llama3 등 대부분의 최신 모델에서 공통적으로 사용되는 모듈
        target_modules=["q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    print("\n--- Trainable Parameters ---")
    peft_model.print_trainable_parameters()
    print("--------------------------\n")

    # 데이터셋 준비
    print("  Preparing dataset...")
    data_files = {"train": config['data']['train_path'],
                  "validation": config['data']['dev_path']}
    dataset = load_dataset("csv", data_files=data_files)

    show_dev_predictions = config.get('show_dev_predictions', False)
    fixed_dev_samples = None
    if show_dev_predictions:
        # 미리보기용 dev 샘플 3개 고정
        fixed_dev_samples = dataset['validation'].shuffle(
            seed=42).select(range(3)).to_pandas()

    use_fewshot = config['prompt'].get('use_fewshot', False)
    num_fewshot = config['prompt'].get('num_fewshot', 1)
    few_shot_disable_chars = config['prompt'].get('few_shot_disable_chars')
    eval_few_shot_samples = None

    if use_fewshot:
        print(f"  Using {num_fewshot}-shot prompt format (dynamic sampling).")
        few_shot_max_chars = config['prompt'].get('few_shot_max_chars')
        few_shot_pool = dataset['train']
        if few_shot_max_chars is not None:
            candidate_pool = dataset['train'].filter(
                lambda x: len(x['dialogue']) <= few_shot_max_chars)
            if candidate_pool.num_rows >= num_fewshot:
                few_shot_pool = candidate_pool
                print(
                    f"  Few-shot pool filtered to <= {few_shot_max_chars} chars (size: {few_shot_pool.num_rows}).")
            else:
                print(
                    f"  Warning: only {candidate_pool.num_rows} samples <= {few_shot_max_chars} chars; using full train set for few-shot examples.")

        base_train_dataset = dataset['train']

        def sample_few_shot_rows(source_dataset, k, seed=None):
            if source_dataset.num_rows < k:
                source_dataset = base_train_dataset
            shuffled = source_dataset.shuffle(
                seed=seed if seed is not None else random.randint(0, 10**6))
            take = min(k, shuffled.num_rows)
            return shuffled.select(range(take))

        def should_use_few_shot_for_dialogue(dialogue_text: str) -> bool:
            if few_shot_disable_chars is None:
                return True
            return len(dialogue_text) <= few_shot_disable_chars

        def format_dataset_dynamically(example, few_shot_dataset):
            dialogue_text = example.get('dialogue', '')
            if not should_use_few_shot_for_dialogue(dialogue_text):
                return {"text": create_prompt(example)}

            selected_rows = sample_few_shot_rows(few_shot_dataset, num_fewshot)
            few_shot_samples = selected_rows.to_pandas()
            return {"text": create_prompt_with_fewshot(example, few_shot_samples=few_shot_samples)}

        prompt_map_function = partial(
            format_dataset_dynamically, few_shot_dataset=few_shot_pool)

        eval_selected_rows = sample_few_shot_rows(
            few_shot_pool, num_fewshot, seed=42)
        eval_few_shot_samples = eval_selected_rows.to_pandas()

        def eval_prompt_map_function(x):
            if not should_use_few_shot_for_dialogue(x.get('dialogue', '')):
                return {"text": create_prompt(x)}
            return {"text": create_prompt_with_fewshot(x, few_shot_samples=eval_few_shot_samples)}
    else:
        print("  Using zero-shot prompt format.")
        def prompt_map_function(x): return {"text": create_prompt(x)}
        eval_prompt_map_function = prompt_map_function

    train_dataset = dataset['train'].map(prompt_map_function)
    eval_dataset = dataset['validation'].map(eval_prompt_map_function)

    #  학습 인자(SFTConfig) 설정
    training_params = config['training'].copy()

    # 최고의 모델을 저장하는 기준은 'final_score'
    training_params['metric_for_best_model'] = "final_score"
    training_params['greater_is_better'] = True  # 점수가 높을수록 좋음
    training_params.setdefault('batch_eval_metrics', True)

    # 최신 TRL 버전은 predict_with_generate 파라미터를 지원하지 않으므로 제거
    training_params.pop('predict_with_generate', None)

    # SFTConfig는 eval_strategy를 사용
    if 'evaluation_strategy' in training_params:
        training_params['eval_strategy'] = training_params.pop(
            'evaluation_strategy')

    # 최대 시퀀스 길이 설정 (SFTConfig의 max_length 사용)
    training_params.setdefault(
        'max_length', config.get('max_seq_length', 2048))

    resume_from_checkpoint = training_params.pop(
        'resume_from_checkpoint', False)

    sft_args = SFTConfig(**training_params)

    # SFTTrainer를 사용한 학습
    metric_fn = build_batch_metrics_fn(tokenizer)
    trainer = SFTTrainer(
        model=peft_model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        compute_metrics=metric_fn,
    )

    # PredictionCallback을 trainer에 추가
    if show_dev_predictions and fixed_dev_samples is not None:
        trainer.add_callback(PredictionCallback(
            trainer,
            tokenizer,
            fixed_dev_samples,
            few_shot_samples=eval_few_shot_samples,
            disable_fewshot_threshold=few_shot_disable_chars,
        ))

    print("  Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 학습된 LoRA 어댑터 저장
    final_model_path = os.path.join(
        config['training']['output_dir'], "final_checkpoint")
    trainer.save_model(final_model_path)
    print(f"  LoRA adapters saved to {final_model_path}")
