# SOLAR Pro QLoRA Fine-tuning Pipeline

SOLAR Pro 모델을 QLoRA로 파인튜닝하여 대화 요약 태스크를 수행하는 파이프라인입니다.

## 주요 특징

- **모델**: Upstage SOLAR Pro (22B 파라미터)
- **기법**: QLoRA (4-bit 양자화 + LoRA Fine-tuning)
- **최적화**: Gradient Checkpointing, PagedAdamW 8-bit
- **배치 추론**: 효율적인 GPU 메모리 관리

## 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 디렉토리 구조

```bash
├── configs/
│   └── config.yaml              # 모델 및 학습 설정
├── src/
│   ├── train_qlora.py           # QLoRA 파인튜닝 스크립트
│   ├── inference.py             # 배치 추론 스크립트
│   ├── prepare_train_data.py    # 데이터 전처리
│   └── analyze_datasets_04.py   # 데이터 분석
├── data/
│   ├── train.csv                # 학습 데이터
│   ├── dev.csv                  # 검증 데이터
│   ├── test.csv                 # 테스트 데이터
│   ├── train_formatted/         # 전처리된 학습 데이터
│   └── dev_formatted/           # 전처리된 검증 데이터
├── output/
│   ├── solar-pro-qlora-summarization/  # 학습된 모델
│   └── submission/              # 제출 파일
├── requirements.txt
└── README.md
```

## 실행 방법

### 1. 데이터 준비

```bash
# src 디렉토리로 이동
cd src

# 학습 데이터를 chat 형식으로 변환
python prepare_train_data.py
```

학습 데이터는 다음과 같은 chat 형식으로 변환됩니다:

```python
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Summarize the following dialogue:\n\n{dialogue}"},
        {"role": "assistant", "content": "{summary}"}
    ]
}
```

### 2. 모델 파인튜닝

```bash
# src 디렉토리에서 실행
python train_qlora.py
```

**주요 설정** (코드 내 수정 가능):

- `LORA_R`: 16 (LoRA rank)
- `LORA_ALPHA`: 32
- `BATCH_SIZE`: 4
- `GRADIENT_ACCUMULATION`: 4 (effective batch size = 16)
- `LEARNING_RATE`: 2e-4
- `NUM_EPOCHS`: 2
- `MAX_SEQ_LENGTH`: 2048

**학습 결과**:

- 모델 저장 위치: `output/solar-pro-qlora-summarization/`
- 중간 체크포인트: `save_steps=50`마다 자동 저장
- 최대 저장 개수: `save_total_limit=3`

### 3. 추론

```bash
# src 디렉토리에서 실행
python inference.py
```

**추론 설정**:

- Batch size: 2 (메모리 효율성)
- Max new tokens: 512
- Decoding: Greedy (do_sample=False)
- 출력 파일: `sample_submission.csv`

## 설정 파일

`configs/config.yaml`에서 다양한 하이퍼파라미터를 조정할 수 있습니다:

```yaml
model:
  model_id: upstage/solar-pro-preview-instruct
  quant: 4bit
  max_memory_gb: 23

generation:
  max_new_tokens: 120
  temperature: 1.0
  repetition_penalty: 1.1
  
prompt:
  user_template: |
    다음 대화를 대화 길이의 20% 정도로 요약하세요.
    대화에 나온 단어를 최대한 활용하세요.
    
    {dialogue}
    
    요약:
```

## 제출 결과

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
|-------|---------|---------|---------|-------------|
| SOLAR Pro (22B) QLoRA | 0.4690 | 0.2910 | 0.3857 | **38.1879** |

## 참고 자료

- [SOLAR Pro 모델](https://huggingface.co/upstage/solar-pro-preview-instruct)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [PEFT 라이브러리](https://github.com/huggingface/peft)
