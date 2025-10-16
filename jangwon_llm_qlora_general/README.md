# Dialogue Summarization

## 파이프라인 소개

본 파이프라인은 QLoRA(Quantized Low-Rank Adaptation) 기술을 활용하여 대규모 언어모델(LLM)을 효율적으로 파인튜닝하고, 일상 대화를 정확하게 요약하는 AI 모델을 구축하는 것을 목표로 합니다.

## 주요 특징

### 1. 효율적인 LLM 파인튜닝

- **QLoRA 기술**: 4-bit 양자화를 통해 메모리 사용량을 대폭 절감하면서도 성능 유지
- **LoRA Adapter**: 전체 모델이 아닌 일부 파라미터만 학습하여 효율성 극대화
- **지원 모델**: SOLAR-10.7B-Instruct, Llama3, Qwen 등 다양한 LLM 지원

### 2. 데이터 증강 파이프라인

- **Paraphrase (재구성)**: LLM을 활용한 대화문 의미 보존 재작성
- **Speaker Swap (화자 교환)**: 대화 순서를 변경하여 다양성 확보
- **Synthetic Generation (합성 데이터)**: 새로운 대화 샘플 생성
- **통합 및 셔플**: 원본 데이터와 증강 데이터를 균형있게 병합

### 3. 통합 워크플로우

- **One-Command Pipeline**: 데이터 증강 → 학습 → 추론까지 단일 명령으로 실행
- **모듈화된 구조**: 각 단계별 독립 실행 및 건너뛰기 가능
- **체크포인트 관리**: 학습 재개 및 최적 모델 자동 저장

## 개발 환경 및 기술 스택

- **언어**: Python 3.10
- **주요 라이브러리**:
  - PyTorch 2.8.0 (CUDA 11.8+)
  - Transformers 4.57.0 (Hugging Face)
  - PEFT 0.17.1 (Parameter-Efficient Fine-Tuning)
  - TRL 0.23.1 (Transformer Reinforcement Learning)
  - BitsAndBytes 0.48.1 (양자화)
  - Pandas 2.1.4
  - NumPy 1.23.5
  - ROUGE 1.0.1 (평가 지표)
- **트래킹 도구**: Weights & Biases (실험 트래킹)
- **버전 관리**: Git / GitHub
- **협업 도구**: GitHub, Notion

## 파이프라인 구조

```bash
├── README.md                    # 파이프라인 개요 및 실행 가이드
├── environment.yml              # Conda 환경 설정 파일
│
├── data/                        # 데이터 디렉토리
│   ├── clean/                   # 전처리된 원본 데이터
│   │   └── denoise/             # 노이즈 제거된 데이터
│   ├── augmented/               # 증강 데이터
│   └── debug/                   # 디버그용 소규모 데이터셋
│
├── docs/                        # 문서 및 참고 자료
│   ├── denoise_rule.xlsx        # 데이터 정제 규칙
│   └── LLM_Pipeline.md          # 파이프라인 상세 설명
│
├── notebooks/                   # Jupyter 노트북
│   ├── EDA.ipynb                # 탐색적 데이터 분석
│   └── EDA_augmented.ipynb      # 증강 데이터 분석
│
├── src/                         # 소스 코드
│   ├── __init__.py
│   ├── main.py                  # 통합 파이프라인 실행 스크립트
│   ├── requirements.txt         # Python 패키지 의존성
│   │
│   ├── data/                    # 데이터 처리 모듈
│   │   ├── augment_data.py      # 데이터 증강 로직
│   │   ├── merge_data.py        # 데이터셋 병합
│   │   ├── dataset_llm.py       # LLM용 Dataset 클래스
│   │   └── preprocess/          # 전처리 모듈
│   │       ├── denoise.py       # 노이즈 제거
│   │       └── typo_fix.py      # 오타 수정
│   │
│   ├── train/                   # 학습 모듈
│   │   └── train_llm.py         # QLoRA 파인튜닝 로직
│   │
│   └── inference/               # 추론 모듈
│       └── inference_llm.py     # 배치 추론 및 제출 파일 생성
│
├── model/                       # 학습된 모델 저장 디렉토리
│   └── *_qlora_fewshot/     # 모델별/설정별 폴더
│       ├── final_checkpoint/    # 최종 LoRA 어댑터
│       └── checkpoint-*/        # 중간 체크포인트
│
└── output/                      # 추론 결과 출력
```

## 빠른 시작 가이드

### 1. 환경 설정

#### 방법 1: Conda 환경 파일 사용 (권장)

```bash
# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate llm-env
```

#### 방법 2: 수동 설치

```bash
# 가상환경 생성
conda create -n llm-env python=3.10
conda activate llm-env

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
pip install -r src/requirements.txt
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 추가합니다:

```bash
# Hugging Face 토큰 (모델 다운로드용)
HF_TOKEN="hf_your_token_here"

# Upstage API 키 (데이터 증강용)
UPSTAGE_API_KEY="sk_your_api_key_here"
```

### 3. 전체 파이프라인 실행

```bash
# 데이터 증강 → 학습 → 추론 전체 실행
python -m src.main
```

### 4. 단계별 실행

```bash
# 증강만 건너뛰고 학습 + 추론
python -m src.main --skip-augmentation

# 학습만 실행 (증강 데이터 이미 존재)
python -m src.main --skip-augmentation --skip-inference

# 추론만 실행 (학습된 모델 사용)
python -m src.main --skip-augmentation --skip-training

# 학습 재개 (체크포인트에서)
python -m src.main --resume
```

### 5. 디버그 모드

```bash
# 소규모 데이터로 빠른 테스트
python -m src.main --debug-small
```

## 주요 기능

### 1. 데이터 증강

- **Paraphrase (재구성)**: LLM이 대화의 의미를 유지하면서 다르게 표현
- **Speaker Swap (화자 교환)**: 대화 순서를 무작위로 섞어 다양한 패턴 학습
- **Synthetic Generation (합성 생성)**: 완전히 새로운 대화 샘플 생성
- **자동 병합**: 원본 + 증강 데이터를 자동으로 통합 및 셔플

### 2. QLoRA 파인튜닝

- **메모리 효율성**: 4-bit 양자화로 GPU 메모리 사용량 최소화
- **LoRA 어댑터**: 전체 모델 대신 작은 어댑터만 학습
- **Few-shot Learning**: 프롬프트에 예시를 포함하여 성능 향상
- **체크포인트 관리**: 자동 저장 및 최적 모델 선택

### 3. 배치 추론 엔진

- **효율적인 배치 처리**: GPU/CPU에 따라 최적화된 배치 크기
- **자동 토큰 관리**: 입력 길이에 따른 동적 처리
- **제출 파일 생성**: 대회 형식에 맞는 CSV 자동 생성

## 실험 설정 커스터마이징

`src/main.py` 파일의 설정을 수정하여 다양한 실험이 가능합니다:

```python
# 모델 선택
model_id = "Upstage/SOLAR-10.7B-Instruct-v1.0"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 학습 하이퍼파라미터
training_config = {
    'num_train_epochs': 3,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
    'lr_scheduler_type': "cosine",
    'warmup_ratio': 0.1,
    ...
}

# LoRA 설정
lora_config = {
    'r': 16,          # LoRA rank
    'alpha': 32,      # LoRA alpha
    'dropout': 0.05,  # Dropout rate
}

# Few-shot 설정
prompt_config = {
    'use_fewshot': True,
    'num_fewshot': 1,
    'few_shot_max_chars': 1024,
}
```

## 평가 지표

본 프로젝트는 ROUGE 점수를 사용하여 요약 품질을 평가합니다:

- **ROUGE-1**: 단어 단위 일치도
- **ROUGE-2**: 2-gram 일치도  
- **ROUGE-L**: 최장 공통 부분 수열 (유창성)

### 제출 결과

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
|-------|---------|---------|---------|-------------|
| Solar 10.7B | 0.5500 | 0.3477 | 0.4357 | **44.4480** |

## 트러블슈팅

### 1. GPU 메모리 부족

**증상**: CUDA out of memory 에러

**해결**:

```bash
# 배치 크기 감소
training_config['per_device_train_batch_size'] = 1
training_config['gradient_accumulation_steps'] = 16

# 최대 시퀀스 길이 감소
llm_config['max_seq_length'] = 2048
```

### 2. 체크포인트 로드 실패

**증상**: `resume_from_checkpoint` 에러

**해결**:

```bash
# 체크포인트 경로 확인
ls model/solar_qlora_fewshot/

# 수동으로 체크포인트 지정
# src/main.py에서 resume_from_checkpoint 값 수정
```

### 3. API 키 인증 오류

**증상**: Hugging Face 또는 Upstage API 에러

**해결**:

```bash
# .env 파일 존재 확인
cat .env

# 토큰 갱신
# https://huggingface.co/settings/tokens
# https://console.upstage.ai/api-keys
```

## 성능 최적화 팁

1. **데이터 증강량 조절**: `--num-paraphrase`, `--num-speaker-swap`, `--num-synthetic` 플래그 활용
2. **Few-shot 예시 최적화**: 프롬프트 길이와 예시 개수 조정
3. **학습률 스케줄링**: Cosine decay로 안정적인 수렴
4. **Early Stopping**: 검증 손실 기반 조기 종료로 과적합 방지
5. **체크포인트 저장**: `save_total_limit`로 디스크 공간 관리

## 참고 자료

- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [PEFT 라이브러리](https://github.com/huggingface/peft)
- [TRL 라이브러리](https://github.com/huggingface/trl)
- [SOLAR 모델](https://huggingface.co/Upstage/SOLAR-10.7B-Instruct-v1.0)
