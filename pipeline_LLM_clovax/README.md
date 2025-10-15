# NLP 프로젝트 템플릿 (HyperCLOVA X 로컬 추론)

이 디렉터리는 HyperCLOVA X 지시형(멀티모달) 모델을 **로컬**로 실행하여 `test.csv`의 요약문과 주제를 생성하는 파이프라인을 제공합니다. 기존 베이스라인 노트북은 참고용으로 남겨 두었고, 실제 결과는 내려받은 Clova X 모델이 직접 생성하도록 구성했습니다.

## 디렉터리 구조

```
NLP/
├── configs/               # 설정 파일 (예: clovax.yaml)
├── data/                  # train/dev/test CSV 및 sample_submission
├── output/
│   ├── model/             # 중간 산출물, few-shot 예시, 세부 결과
│   └── submission/        # 제출용 CSV 저장 위치
└── src/
    ├── inference/         # Clova X 로컬 추론 파이프라인
    ├── utils/             # 설정, 입출력, few-shot 보조 함수
    └── main.py            # 기본 진입점 (run_pipeline 래핑)
```

## 사용 방법

1. **모델 준비**  
   Hugging Face에서 HyperCLOVA X 지시형 텍스트 모델 체크포인트를 내려받습니다. `configs/clovax.yaml`의 `model.local_dir`에 로컬 경로를 지정하거나, Hugging Face 캐시에 이미 내려받은 경우 자동으로 감지합니다. 접근 권한이 있다면 `model.model_id`를 그대로 사용해 온라인에서도 로드할 수 있습니다. 4bit/8bit 양자화를 쓰려면 CUDA + `bitsandbytes`가 지원되는 GPU 환경이 필요합니다.

2. **설정 점검**  
   - `model.local_dir`가 비어 있으면 Hugging Face 캐시(`~/.cache/huggingface/hub`)를 자동으로 스캔합니다.
   - `model.token`에 Hugging Face 액세스 토큰을 지정하면 게이트된 저장소도 다운로드할 수 있습니다.
   `configs/clovax.yaml`에서 데이터/출력 경로, 모델 ID·양자화 옵션, 생성 하이퍼파라미터, few-shot 샘플링 전략 등을 원하는 대로 조정합니다.

3. **파이프라인 실행** (처음 실행 시 모델을 자동으로 감지하거나 다운로드)
   ```bash
   cd NLP
   python -m src.inference.inference --cfg configs/clovax.yaml
   # 또는
   python -m src.main
   ```

4. **결과물 확인**
   - `output/model/clovax_generations.csv`: fname, summary, topic, 원본 응답을 포함한 세부 결과
   - `output/submission/clovax_submission.csv`: `sample_submission.csv` 형식에 맞춘 제출 파일
   - `output/model/fewshot_examples.jsonl`: 사용된 few-shot 예시(선택 저장)

## 주요 모듈

- `src/utils/config.py`: YAML 로더 + `${VAR:default}` 환경변수 치환
- `src/utils/examples.py`: `train.csv`에서 few-shot 예시 샘플링
- `src/inference/prompt_builder.py`: 시스템/사용자/예시 메시지를 조합해 챗 포맷 생성
- `src/inference/local_clovax.py`: HyperCLOVA X 로컬 모델 로더 및 텍스트 생성기
- `src/inference/inference.py`: 데이터 로딩 → few-shot 선정 → 모델 추론 → CSV 저장 전체 파이프라인

## 참고 사항

- `baseline.ipynb`는 원래 대회 베이스라인을 이해하기 위한 참고용입니다. 현재 파이프라인은 전처리·학습 없이 로컬 Clova X로 직접 요약/주제 생성을 수행합니다.
- Clova X 응답이 JSON 형식을 벗어나는 경우를 대비해 간단한 파싱 로직을 포함해 두었습니다. 필요에 따라 강화하세요.
- 모델 용량이 크므로 디스크와 VRAM을 충분히 확보해야 합니다. 4bit/8bit 양자화로 메모리 사용량을 줄일 수 있습니다.
- 주제 라벨은 `train.csv`의 `topic` 컬럼 스타일을 따르도록 프롬프트를 작성했습니다. 필요한 경우 허용 라벨 목록을 직접 정의해 보완해도 좋습니다.
