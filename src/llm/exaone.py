from transformers import TrainingArguments as HFTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
import evaluate
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from src.data.preprocess.preprocess import make_input_exaone


from src.utils.gpu_utils import clean_gpu_memory
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_and_train_exaone():
    
    # GPU 초기화
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 데이터셋 로드 (당신의 대화 요약 데이터)
    train_df = pd.read_csv("/workspace/NLP_Dialogue_Summarization/data/train.csv")
    val_df = pd.read_csv("/workspace/NLP_Dialogue_Summarization/data/dev.csv")  # 있다면

    train_texts = make_input_exaone(train_df, is_test=False)
    val_texts = make_input_exaone(val_df, is_test=False)

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # ⭐ 이 부분 확인해보세요!
    print(f"Dataset 타입: {type(train_dataset)}")
    print(f"Dataset[0]: {train_dataset[0]}")  # {'text': '...'} 이어야 함
    print(f"Dataset[0]['text'] 타입: {type(train_dataset[0]['text'])}")

    
    # 1. 모델과 토크나이저 로드 (4bit 양자화)
    model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ## 2. Special tokens 추가
    special_tokens_dict = {
        "additional_special_tokens": [
            "<topic>", "</topic>", 
            "<dialogue>", "</dialogue>"
        ]
    }
    num_added_special = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_special} special tokens")

    # 3. Basic tokens 추가 
    basic_tokens = [
        "#Person1#", "#Person2#", "#Person3#", 
        "#PhoneNumber#", "#Address#", "#PassportNumber#", 
        "#Email#", "#Date#", "#Time#", 
        "#Number#", "#Money#", 
        "#Organization#", "#Location#"
    ]
    num_added_basic = tokenizer.add_tokens(basic_tokens)
    print(f"Added {num_added_basic} basic tokens")

    # 4. tokenizer 설정 : pad_token , padding (Causal LM은 보통 eos를 pad로 씀)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 512  #  길이 강제    

    print(f"Total vocabulary size: {len(tokenizer)}")
    
    # 5. 모델 로드
    qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # 3090이면 fp16이 안전
            )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=qconf,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # 토큰 추가했으니 모델도 확장
    model.resize_token_embeddings(len(tokenizer))
    
        
    # 토크나이저 테스트
    print("="*80)
    print("3. 토크나이저 테스트")
    print("="*80)
    try:
        sample_text = train_texts[0]
        tokens = tokenizer(sample_text, truncation=True, max_length=512, padding="max_length")
        print(f"Tokenized keys: {tokens.keys()}")
        print(f"input_ids type: {type(tokens['input_ids'])}")
        print(f"input_ids length: {len(tokens['input_ids'])}")
        print("✓ 토크나이저 정상 작동")
    except Exception as e:
        print(f"✗ 토크나이저 에러: {e}")
        import traceback
        traceback.print_exc()

    
    # 메모리 절약 설정
    model.config.use_cache = False
    model.config.attn_implementation = "sdpa" 

    # 6. LoRA 설정 (빠른 학습)
    lora_config = LoraConfig(
        r=4,  # 랭크 (낮을수록 빠름) 16→8로 낮춰 VRAM 절약
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # 일부만 학습
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    # 7. 학습 설정 (RTX 3090 최적화)
    sft_config = SFTConfig(
    output_dir="/workspace/NLP_Dialogue_Summarization/output/model/exaone/",
    
    # === 배치 설정 ===
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    
    # === 학습 설정 ===
    num_train_epochs=1,
    learning_rate=2e-4,
    max_steps=500,
    
    # === SFTConfig 전용 설정 ===
    dataset_text_field="text",  #  여기로 이동
    max_seq_length=512,         #  여기로 이동
    packing=False,              #  여기로 이동
    
    # === 메모리 절약 ===
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # === 로깅 & 저장 ===
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    eval_strategy="no",
    save_strategy="steps",
    
    # === 기타 ===
    remove_unused_columns=True,  #  True로 변경!
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    report_to="none",
)
    
    # ===== 메모리 콜백 =====
    class MemoryCleaner(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                torch.cuda.empty_cache()
        
        def on_evaluate(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
        
    # 7. 학습 설정 (RTX 3090 최적화) 
    training_args = HFTrainingArguments( 
        output_dir="/workspace/NLP_Dialogue_Summarization/output/model/exaone/",
        
        # === 학습 설정 ===
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        
         # === 평가 설정 추가! ===
        per_device_eval_batch_size=1,  # ✅ 평가 배치도 1로! (매우 중요)
        eval_accumulation_steps=2,  # 메모리 절약용
            
        gradient_checkpointing=True,            # ✅ 핵심
        optim="paged_adamw_8bit",               # ✅ 핵심(메모리↓)
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        save_strategy="epoch",
        eval_strategy="epoch",
        # eval_strategy="no",  #평가 비활성화
        
        # 3.  체크포인트 개수 제한
        save_steps=500,  # 더 자주 저장
        save_total_limit=3,  # 최근 3개만 보관 (디스크 절약)
        ignore_data_skip=True,
        # 4.  최고 성능 모델만 저장
        load_best_model_at_end=True,  # 학습 끝날 때 베스트 모델 로드
        metric_for_best_model="eval_rougeL",  # 어떤 지표 기준? (loss, accuracy 등)
        greater_is_better=True,
        # 5.  최종 모델 저장 여부
        save_safetensors=True,  # safetensors 형식 (권장)
        
        # === 기타 ===
        remove_unused_columns=False,
        dataloader_pin_memory=False,  #  메모리 이슈 해결
        dataloader_num_workers=0,  #  메모리 절약
    )

    rouge = evaluate.load("rouge")
    def compute_metrics_rouge(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds); labels = np.asarray(labels)

        # logits → token id
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        # -100 → pad로 교체 후 디코딩
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        labels = np.where(labels == -100, pad_id, labels)

        pred_texts  = tokenizer.batch_decode(preds.tolist(),  skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        def pp(xs):  # 선택: 특수토큰 정리
            return [x.replace("<dialogue>", "").replace("</dialogue>", "").strip() for x in xs]

        scores = rouge.compute(predictions=pp(pred_texts), references=pp(label_texts), use_stemmer=True)
        # ✅ float로 바로 들어옴: mid.fmeasure 쓰지 말고 그대로 반환
        return {
            "eval_rouge1": float(scores["rouge1"]),
            "eval_rouge2": float(scores["rouge2"]),
            "eval_rougeL": float(scores["rougeL"]),
            # 필요하면 "eval_rougeLsum": float(scores["rougeLsum"]),
        }
            
    # def fmt(example):
    #     # Dataset가 {"text": "..."} 구조면 그대로 반환
    #     return example["text"]    

    # 8. SFTTrainer로 간단하게!
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 옵션
        dataset_text_field="text",
        max_seq_length=512,      
        tokenizer=tokenizer,       
        
        # args=training_args,
        args=sft_config,
        
        peft_config=lora_config,
        compute_metrics=compute_metrics_rouge,  
        packing=False,         # RTX 3090은 패킹 비활성화 (메모리 부족 방지)
        callbacks=[MemoryCleaner()],  # 콜백 추가
    )

    # 9. 학습 시작
    trainer.train()  # RTX 3090: 약 3-4시간 예상
    trainer.save_model("/workspace/NLP_Dialogue_Summarization/output/model/exaone/best_model")
    tokenizer.save_pretrained("/workspace/NLP_Dialogue_Summarization/output/model/exaone/best_model")
    
    # 메모리 정리
    torch.cuda.empty_cache()
    print("완료!")
    







def load_and_infer_exaone():
    # 1) 경로/모델
    MODEL_DIR = "/workspace/NLP_Dialogue_Summarization/output/model/exaone/best_model"
    TEST_CSV = "/workspace/NLP_Dialogue_Summarization/data/test_topic_solar.csv"
    OUT_CSV  = "/workspace/NLP_Dialogue_Summarization/output/submission/submission.csv"

    # 2) 데이터 로드 (test.csv에는 최소 'id','dialogue','topic' 컬럼 필요)
    test_df = pd.read_csv(TEST_CSV)

    # 3) 프롬프트 생성
    test_texts = make_input_exaone(test_df, is_test=True)
    test_ds = Dataset.from_dict({"text": test_texts})

    # 4) 토크나이저/모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("모델 로드 중...")
    # ⭐ 방법 1: 베이스 모델 + LoRA 어댑터 로드 (권장)
    base_model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,  # torch_dtype 대신 dtype 사용
        device_map="auto",
        trust_remote_code=True,
    )
    
    # ⭐ 토큰 임베딩 크기 조정 (중요!)
    model.resize_token_embeddings(len(tokenizer))
    
    # ⭐ LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    model = model.merge_and_unload()  # LoRA 머지 (선택사항: 추론 속도 향상)
    
    model.eval()
    print("모델 로드 완료!\n")

    # 5) 배치 생성 함수
    def generate_batch(batch):
        enc = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,      # 대회 제출용이면 보통 deterministic
                temperature=0.7,      # do_sample=True로 바꾸면 사용
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        # 6) "### Output:" 이후만 요약으로 추출
        preds = []
        for d in decoded:
            if "### Output:" in d:
                pred = d.split("### Output:")[-1].strip()
            else:
                pred = d.strip()
            preds.append(pred)
        return {"pred": preds}

    # 7) 배치 추론
    print("추론 중...")
    pred_ds = test_ds.map(generate_batch, batched=True, batch_size=4)
    print("추론 완료!\n")

    # 8) 제출 파일 저장
    submission = pd.DataFrame({
        "fname": test_df["fname"],
        "summary": pred_ds["pred"],
    })
    submission.to_csv(OUT_CSV, index=False)
    print(f"✓ 저장 완료 → {OUT_CSV}")
    print(f"✓ 샘플:\n{submission.head()}")
