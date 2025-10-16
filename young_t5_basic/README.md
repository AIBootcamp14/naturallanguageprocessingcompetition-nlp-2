# T5 Basic Pipeline

pko-t5-large 모델을 Fine-tuning하여 대화 요약 태스크를 수행하는 기본 파이프라인입니다.

## 주요 특징

- **모델**: paust/pko-t5-large (한국어 특화 T5 모델)
- **아키텍처**: Encoder-Decoder (Seq2Seq)
- **학습 방식**: Full Fine-tuning
- **평가 지표**: ROUGE-1, ROUGE-2, ROUGE-L F1 Score

## 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 디렉토리 구조

```bash
young_t5_basic/
├── configs/
│   └── config.yaml          # 모든 설정 (모델, 학습, 추론)
├── src/
│   ├── main.py              # 통합 실행 스크립트
│   ├── data_preparation.py  # 데이터 준비
│   ├── create_submission.py # 제출 파일 생성
│   └── full_eda.py          # EDA 분석
├── data/
│   ├── train.csv            # 학습 데이터
│   ├── dev.csv              # 검증 데이터
│   └── test.csv             # 테스트 데이터
├── logs/                    # 학습 로그
├── checkpoint-*/            # 학습된 체크포인트
├── prediction/              # 추론 결과
├── requirements.txt
└── README.md
```

## 설정 파일 (configs/config.yaml)

모든 하이퍼파라미터는 `configs/config.yaml`에서 관리됩니다:

```yaml
general:
  data_path: ./data/
  model_name: paust/pko-t5-large
  output_dir: ./

tokenizer:
  encoder_max_len: 1024      # 대화 입력 최대 길이
  decoder_max_len: 150       # 요약 출력 최대 길이
  special_tokens:            # 추가 특수 토큰
    - '#Person1#'
    - '#Person2#'

training:
  do_train: true             # 학습 실행 여부
  num_train_epochs: 20
  learning_rate: 1.0e-05
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_ratio: 0.1
  fp16: true
  gradient_checkpointing: true

inference:
  batch_size: 8
  generate_max_length: 90
  num_beams: 4
  no_repeat_ngram_size: 3
```

## 실행 방법

### 1. 학습

```bash
# configs/config.yaml에서 do_train: true로 설정
cd src
python main.py
```

**학습 설정**:

- Epochs: 20
- Batch size: 4 (effective: 16 with gradient accumulation)
- Learning rate: 1e-5
- Scheduler: Cosine with warmup
- Early stopping: Patience 3, threshold 0.001
- 최적 모델 자동 저장

### 2. 추론

학습 완료 후, 자동으로 최신 체크포인트로 추론이 진행됩니다.

```bash
# src/main.py 하단 설정 수정
loaded_config['training']['do_train'] = False
loaded_config['inference']['ckt_path'] = './checkpoint-10123'

# src 디렉토리에서 실행
cd src
python main.py
```

**추론 설정**:

- `is_final_submission=True`: test.csv 사용 → prediction.csv 생성  
- `is_final_submission=False`: dev.csv 사용 → output.csv 생성 + 로컬 ROUGE 계산

### 3. 로컬 검증

Dev 데이터셋으로 추론 시 자동으로 ROUGE 점수를 계산합니다:

```python
output = inference(loaded_config, is_final_submission=False)
calculate_local_rouge(output, dev_data_path)
```

출력 예시:

```bash
✨✨✨ 로컬 검증 (Dev Set) ROUGE F1 점수 ✨✨✨
ROUGE-1 F1 평균: 57.5000
ROUGE-2 F1 평균: 37.6600
ROUGE-L F1 평균: 48.8100
➡️ 최종 로컬 평균 F1 점수: 47.9900
```

## 주요 기능

### 데이터 전처리

- **Preprocess 클래스**: 대화문에서 노이즈 제거 (`\n`, `<br>` 등)
- **Special Tokens**: `#Person1#`, `#Person2#` 등 화자 정보 보존
- **BOS/EOS 토큰**: 디코더 입력/출력에 자동 추가

### 학습 최적화

- **FP16**: Mixed precision training으로 학습 속도 향상
- **Gradient Checkpointing**: 메모리 효율성 증대
- **Early Stopping**: 과적합 방지
- **Best Model Selection**: Validation loss 기준 최적 모델 자동 저장

### 추론 최적화

- **Beam Search**: num_beams=4로 다양한 후보 탐색
- **No Repeat N-gram**: 반복 문구 방지 (n=3)
- **배치 추론**: GPU 효율적 활용
- **토큰 정제**: `<usr>`, `</s>`, `<pad>` 등 불필요 토큰 제거

## 제출 결과

| Model  | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
|-------|---------|---------|---------|-------------|
| T5 | 0.5750 | 0.3766 | 0.4881 | **47.9550** |

## 파일 설명

- `src/main.py`: 학습/추론 통합 스크립트 (867줄, 모든 기능 포함)
- `configs/config.yaml`: 하이퍼파라미터 설정
- `src/data_preparation.py`: 데이터 로드 및 전처리
- `src/create_submission.py`: 제출 파일 포맷 변환
- `src/full_eda.py`: 탐색적 데이터 분석

## 참고 사항

- 모델 로드 시 `safe_serialization=False` 옵션 제거로 호환성 향상
- 체크포인트는 epoch마다 자동 저장 (`save_total_limit=5`)
- 추론 결과는 `prediction/` 폴더에 저장

## 참고 자료

- [pko-t5-large 모델](https://huggingface.co/paust/pko-t5-large)
- [T5 논문](https://arxiv.org/abs/1910.10683)
