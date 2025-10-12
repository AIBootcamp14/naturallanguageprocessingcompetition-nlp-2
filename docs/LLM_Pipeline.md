# NLP 경진대회: LLM QLoRA Fine-Tuning Workflow Guideline

이 문서는 QLoRA 기술을 사용하여 거대 언어 모델(LLM)을 파인튜닝하는 워크플로우를 안내합니다.

## 1. 환경 설정

**environment.yml 파일을 프로젝트의 메인 아래에 바로 첨부해 두었으니 해당 파일을 이용하여 환경을 설정하시는 것을 추천합니다.**  
**이 경우 1.1 과 1.2 은 스킵하시고 1.3 부터 진행해주세요.**

### 1.1. 가상환경 생성 및 활성화

```bash
# 새 가상환경 생성 (Python 3.10)
conda create -n llm-env python=3.10

# 가상환경 활성화
conda activate llm-env
```

### 1.2. 필수 라이브러리 설치

PyTorch를 먼저 설치한 후, 파인튜닝 및 평가에 필요한 모든 라이브러리를 설치합니다.

혹은 소스코드에서 실제로 사용하는 패키지들은 `src/requirements.txt` 로 저장해두었으니 이를 이용하여 설치하셔도 됩니다.

```bash
# 1. PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. LLM 파인튜닝 및 평가 관련 라이브러리 설치
pip install transformers accelerate bitsandbytes peft trl python-dotenv pandas datasets openai sacremoses python-mecab-ko rouge_score evaluate
```

### 1.3. API 키 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 아래 내용을 추가합니다.

```bash
# .env

# Hugging Face 접근을 위한 토큰 (Llama3, Solar 등 모델 다운로드 시 필수)
HF_TOKEN="hf_...여기에_발급받은_토큰을_입력하세요..."

# 데이터 증강 시 사용할 Upstage API 키
UPSTAGE_API_KEY="sk-...여기에_발급받은_API_키를_입력하세요..."
```

-----

## 2. 통합 파이프라인 실행 (`main.py`)

`src/main.py`는 데이터 증강부터 학습, 추론까지 모든 단계를 제어하는 모듈입니다. 전체 워크플로우를 처음부터 실행할 때는 이 스크립트를 사용하는 것을 권장합니다.

### 2.1. 전체 파이프라인 실행

```bash
# llm-env 가상환경에서 실행
python -m src.main
```

- 추가 옵션: '--num-paraphrase'(Solar paraphrase), '--num-speaker-swap', '--num-synthetic', '--merge-base-path'

증강 후 `data/augmented/`에는 `train_augmented_rp.csv`, `train_augmented_ss.csv`, `train_augmented_synthetic.csv`, `train_augmented.csv`가 생성됩니다.

**참고:** `src/main.py` 파일 하단의 `if __name__ == "__main__":` 블록에서 모델, 데이터 경로, 하이퍼파라미터 등 주요 실험 설정을 손쉽게 변경할 수 있습니다.

### 2.2. 특정 단계 건너뛰기

이미 특정 단계를 완료한 경우, `--skip` 플래그를 사용하여 시간을 절약할 수 있습니다.

```bash
# 예시 1: 데이터 증강을 건너뛰고, 이미 생성된 데이터로 학습 및 추론 실행
python -m src.main --skip-augmentation

# 예시 2: 증강 및 학습을 건너뛰고, 이미 학습된 모델로 추론만 실행
python -m src.main --skip-augmentation --skip-training
```

### 3. 학습 재개/추가 학습 시 체크포인트 재시작

- `python -m src.main --resume`를 사용하면 `model/` 하위 `output_dir`에 저장된 가장 최근 `checkpoint-#####`에서 옵티마이저/스케줄러 상태까지 복원하여 학습을 이어갈 수 있습니다.  
  - 이어서 학습할 때는 `src/main.py`의 `num_train_epochs` 값을 원하는 총 epoch 수(예: 기존 3 → 4)로 늘리면, 직전 학습이 멈춘 지점부터 추가 epoch이 진행됩니다.  
- `--resume` 플래그를 생략하면 `resume_from_checkpoint=False`로 동작하므로 기존과 동일하게 처음부터 학습이 시작됩니다.  
- 체크포인트를 여러 개 보관하려면 `training_config['save_total_limit']` 값을 늘리거나 `None`으로 설정해 자동 삭제를 막아 주세요. 기본값(3)인 상태에서 더 많은 epoch을 돌리면 가장 오래된 체크포인트가 순차적으로 제거됩니다.  
