# T5 with Topic Information Pipeline

Topic 정보를 활용한 T5 대화 요약 파이프라인입니다. 대화문에 주제를 명시적으로 추가하여 요약의 방향성과 초점을 개선합니다.

## 주요 특징

- **모델**: paust/pko-t5-large (한국어 특화 T5 모델)
- **핵심 기법**: Topic-aware Summarization
- **입력 형식**: `<topic> {topic} </topic> <dialogue> {dialogue} </dialogue>`
- **성능 향상**: Topic 추가로 ROUGE Final Score 0.51 상승

## Topic 정보의 효과

Topic을 명시하면 요약이 대화의 핵심 주제에 더 집중하게 되어 정확도가 향상됩니다.

## 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 디렉토리 구조

```bash
yoon_t5_topic/
├── configs/
│   └── config.yaml          # 전체 설정 파일
├── src/
│   ├── main.py              # 실행 진입점
│   ├── data/                # 데이터 로딩 및 전처리
│   ├── model/               # 모델 로딩
│   ├── train/               # 학습 로직
│   ├── inference/           # 추론 로직
│   ├── evaluate/            # 평가 메트릭
│   ├── llm/                 # LLM 관련 (EXAONE 등)
│   └── utils/               # 유틸리티 함수
├── requirements.txt
└── README.md
```

## 설정 파일 (configs/config.yaml)

```yaml
general:
  model_name: paust/pko-t5-large
  output_dir: /path/to/output/model

tokenizer:
  encoder_max_len: 768       # 입력 최대 길이
  decoder_max_len: 128       # 출력 최대 길이
  special_tokens:            # Topic 구분 토큰 추가
    - '<topic>'
    - '</topic>'
    - '<dialogue>'
    - '</dialogue>'

training:
  do_train: true
  num_train_epochs: 35
  learning_rate: 3.5e-05     # T5 권장 범위 (3e-5 ~ 1e-4)
  per_device_train_batch_size: 3
  gradient_accumulation_steps: 32
  optim: adafactor            # T5 권장 옵티마이저
  lr_scheduler_type: polynomial
  bf16: true                  # T5에 최적화
  gradient_checkpointing: true
  label_smoothing_factor: 0.1

inference:
  batch_size: 18
  generate_max_length: 96
  num_beams: 6
  no_repeat_ngram_size: 4
  length_penalty: 1.2
  repetition_penalty: 1.2
  min_length: 25
```

## 실행 방법

### 1. Topic 데이터 준비

대화문에 Topic을 추가한 형식으로 데이터를 준비합니다:

```python
# 입력 형식
input_text = f"<topic> {topic} </topic> <dialogue> {dialogue} </dialogue>"

# 예시
input_text = "<topic> 공예품 쇼핑 </topic> <dialogue> #Person1#: 중국 공예품을 보고 싶어요... </dialogue>"
```

Topic은 Solar Pro API 등을 통해 자동으로 추출할 수 있습니다.

### 2. 학습

```bash
# src/ 디렉토리에서 실행
cd src
python main.py train
```

**주요 학습 설정**:

- Optimizer: Adafactor (T5 공식 권장)
- Scheduler: Polynomial decay
- BF16: Mixed precision (FP16 대신)
- Gradient checkpointing: 메모리 절약
- Label smoothing: 0.1 (과적합 방지)

### 3. 추론

```bash
# src/ 디렉토리에서 실행
cd src
python main.py infer
```

추론 시 최적 체크포인트 경로를 지정합니다:

```python
# main.py의 run_infer 함수에서
ckt_path = "/workspace/NLP_Dialogue_Summarization/output/model/best_model"
config['inference']['ckt_path'] = ckt_path
```

## Topic 정보 활용 예시

### 예시 1: 공예품 쇼핑

**Topic 없이**:
> #Person1#은 중국의 손재주를 보고 싶어합니다. #Person2#는 독특한 중국식으로 운반이 쉬운 종이 공예품, 자수, 바틱을 추천합니다.

**Topic과 함께** (주제: 공예품):
> #Person2#는 중국의 손재주가 뛰어난 <span style="color:red">공예품</span>을 사려고 합니다. #Person1#은 독특한 중국식이면서 운반이 쉬운 종이 공예품, 자수, 바틱을 추천합니다.

→ 주제가 명확해지고 핵심 내용이 강조됩니다.

### 예시 2: 알레르기 식단 관리

**Topic 없이**:
> Sue는 Bill의 생일 파티에 케이크를 가져오지 못해 아쉬워한다. Bill은 Sue에게 샐러드를 가져다주고, Sue가 좋아하는 뜨거운 스프를 가져다주겠다고 제안한다.

**Topic과 함께** (주제: 알레르기 식단 관리):
> Sue는 Bill에게 자신의 생일 파티에 케이크를 가져오라고 제안한다. Bill은 <span style="color:red">알레르기 식단 관리</span>를 위해 특정 음식을 피하고 있으며, Sue가 가져온 샌드위치도 먹지 못했다.

→ 대화의 핵심 맥락("왜")이 정확하게 요약됩니다.

## 최적화 전략

### T5 특화 설정

1. **Adafactor Optimizer**: AdamW 대신 Adafactor 사용 (메모리 효율)
2. **BF16**: FP16 대신 BF16 사용 (T5에 더 안정적)
3. **Polynomial LR Scheduler**: Cosine 대신 Polynomial decay
4. **높은 Learning Rate**: 3.5e-5 (T5 권장 범위)

### 생성 품질 향상

1. **Beam Search**: num_beams=6 (더 많은 후보 탐색)
2. **Length Penalty**: 1.2 (적절한 길이 유도)
3. **Repetition Penalty**: 1.2 (반복 억제)
4. **No Repeat N-gram**: 4 (더 강한 반복 방지)
5. **Min Length**: 25 (너무 짧은 요약 방지)

## 결과

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
|--------|---------|---------|---------|-------------|
| T5 (+ Topic) | 0.5731 | 0.3642 | 0.4850 | **47.4114** |

## Topic 검증

Topic의 품질을 검증하는 유틸리티 함수가 포함되어 있습니다:

```python
# main.py에서
invalid_topics = find_invalid_topics(df, max_topic_len=20, min_confidence=0.7)
```

- Topic 길이가 너무 긴 경우 탐지
- Topic confidence가 낮은 경우 경고
- 비정상 Topic 자동 필터링

## 참고 사항

- Topic 추출에는 Solar Pro API 또는 다른 LLM 사용 가능
- Topic 추가로 인한 입력 길이 증가를 고려해 `encoder_max_len` 조정
- WandB 연동으로 학습 과정 실시간 모니터링 가능

## 참고 자료

- [pko-t5-large 모델](https://huggingface.co/paust/pko-t5-large)
- [T5 논문](https://arxiv.org/abs/1910.10683)
- [Adafactor Optimizer](https://arxiv.org/abs/1804.04235)
