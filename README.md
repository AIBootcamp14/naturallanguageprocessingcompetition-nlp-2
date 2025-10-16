# Dialogue Summarization

일상 대화를 바탕으로 요약문을 생성하는 NLP 경진대회 프로젝트

## 📚 Table of Contents
- [팀 구성원](#팀-구성원)
- [0. Overview](#0-overview)
- [1. 프로젝트 구조](#1-프로젝트-구조)
- [2. 협업 방식](#2-협업-방식)
- [3. EDA](#3-eda)
- [4. 데이터 전처리](#4-데이터-전처리)
- [5. 데이터 증강](#5-데이터-증강)
- [6. 파이프라인별 모델링 및 실험 결과](#6-파이프라인별-모델링-및-실험-결과)
- [7. 리더보드](#7-리더보드)
- [8. 실행 가이드](#8-실행-가이드)
- [9. 회고](#9-회고)
- [10. 발표 자료](#10-발표-자료)
- [11. 참고 자료](#11-참고-자료)

## 팀 구성원

| ![김장원](https://avatars.githubusercontent.com/u/128503571?v=4&s=200) | ![김영](https://avatars.githubusercontent.com/u/213391898?v=4&s=200) | ![문채린](https://avatars.githubusercontent.com/u/213385368?s=200&u=199e83da989abfc5387e2b64c00751a77bb5c6cc&v=4) | ![민병호](https://avatars.githubusercontent.com/u/213389909?s=200&u=637057beaf59c03a304331ca2c5838c029195669&v=4) | ![이윤서](https://avatars.githubusercontent.com/u/77047118?s=200&v=4) | ![정민지](https://avatars.githubusercontent.com/u/208557619?s=200&v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [![GitHub](https://img.shields.io/badge/GitHub-김장원👑-181717?style=&logo=github&logoColor=white)](https://github.com/jkim1209)          |            [![GitHub](https://img.shields.io/badge/GitHub-김영-181717?style=flat&logo=github&logoColor=white)](https://github.com/kimyoung9689)            |            [![GitHub](https://img.shields.io/badge/GitHub-문채린-181717?style=flat&logo=github&logoColor=white)](https://github.com/CHAERINMOON)             |            [![GitHub](https://img.shields.io/badge/GitHub-민병호-181717?style=flat&logo=github&logoColor=white)](https://github.com/BH-Min-lab)              |            [![GitHub](https://img.shields.io/badge/GitHub-이윤서-181717?style=flat&logo=github&logoColor=white)](https://github.com/riicoseo)              |          [![GitHub](https://img.shields.io/badge/GitHub-정민지-181717?style=flat&logo=github&logoColor=white)](https://github.com/mingg210)            |
|                팀장, 데이터 전처리 · 증강 · LLM 모델링                   |                    데이터 전처리 · T5 모델링                           |                    데이터 전처리 · 모델링                               |                    EDA · LLM 모델링                        |                    데이터 전처리 · T5 모델링                         |                    데이터 전처리 · 모델링                               |

## 0. Overview

### 프로젝트 소개  

일상생활에서 대화는 항상 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.

**본 프로젝트는 학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 중 하는 대화들에 대해 정확하고 간결한 요약을 생성하는 AI 모델을 구축합니다.**

### 대회 정보

- **기간**: 2025년 9월 26일 ~ 2025년 10월 16일
- **주제**: Dialogue Summarization (일상 대화 요약)
- **평가 지표**: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L의 F1 Score 평균)

$$
\text{Final Score} = \frac{ \text{ROUGE-1\\_F1} + \text{ROUGE-2\\_F1} + \text{ROUGE-L\\_F1} }{3}
$$


- **목표**:
  - 다양한 데이터 증강 기법 시도
  - Encoder-Decoder (BART, T5) 및 Decoder-only (LLM) 아키텍처 비교
  - QLoRA를 활용한 효율적인 LLM Fine-tuning

### 환경

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=plastic&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=plastic&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.x-FF6F00?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=plastic&logo=nvidia&logoColor=white)

- **OS**: Ubuntu 20.04 LTS
- **Python**: 3.10
- **PyTorch**: 2.x (파이프라인별로 상이)
- **Transformers**: 4.x (파이프라인별로 상이)
- **CUDA**: 11.8+
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)

## 1. 프로젝트 구조

<details>
<summary><b>📁 최상위 프로젝트 구조 보기</b></summary>

<br>

> 본 프로젝트는 5개의 독립적인 파이프라인으로 구성되어 있으며,  
> 각 폴더는 서로 다른 모델 아키텍처를 기반으로 합니다.

```bash
NLI-Dialogue-Summarization/
├── README.md                        # 프로젝트 전체 개요
├── .gitignore                       # Git 추적 제외 파일
│
├── data/                            # 공통 데이터 (gitignored)
│
├── bh_llm_clovax_api/              # HyperCLOVA X API 파이프라인
├── bh_llm_qlora_solar/             # Solar Pro 특화 QLoRA 파이프라인
├── jangwon_llm_qlora_general/      # LLM 기본 QLoRA 파이프라인
├── young_t5_basic/                 # 기본 T5 파이프라인
└── yoon_t5_topic/                  # Topic 활용한 T5 파이프라인
```
</details>

### 파이프라인 개요

본 프로젝트는 5개의 독립적인 파이프라인으로 구성되어 있으며, 각 파이프라인은 서로 다른 모델과 접근 방식을 사용합니다.

| 파이프라인 | 모델 아키텍처 | 주요 기법 | 주 담당자 | 상세 가이드 |
|-----------|--------------|----------|--------|-----------|
| **T5 Basic** | T5 | Seq2Seq Fine-tuning | 김영 | [📁 상세보기](./young_t5_basic/README.md) |
| **T5 Topic** | T5 | Seq2Seq Fine-tuning <br> Topic 추가 활용 | 이윤서 | [📁 상세보기](./yoon_t5_topic/README.md) |
| **LLM QLoRA General** | LLM Model <br> (default: Solar) | QLoRA Fine-tuning | 김장원 | [📁 상세보기](./jangwon_llm_qlora_general/README.md) |
| **LLM Clova X API** | HyperCLOVA X | API 기반 요약 (학습 X) | 민병호 | [📁 상세보기](./bh_llm_clovax_api/README.md) |
| **LLM QLoRA Solar** | SOLAR Pro | QLoRA Fine-tuning | 민병호 | [📁 상세보기](./bh_llm_qlora_solar/README.md) |

각 파이프라인의 실행 방법은 해당 폴더의 README.md를 참고해주세요.

## 2. 협업 방식

- **협업 도구**: GitHub, Slack, Notion
- **회의 일정**: 매일 오전 10시, 오후 6시 정기 회의
- **프로젝트 관리**: GitHub Projects를 활용한 일정 및 이슈 트래킹
- **코드 공유**: GitHub을 통한 버전 관리 및 협업
- **진행 상황 공유**: Slack을 통한 실시간 소통 및 팔로우업

## 3. EDA

### Train Dataset 분석

<p align="center"><img src="https://github.com/user-attachments/assets/3a37878d-9ce7-4683-b4f0-6cffec98cb28" width="90%"></p>

<p align="center"><img src="https://github.com/user-attachments/assets/3489dc7f-cbed-4904-9c43-f2466df89640" width="90%"></p>

### Validation Dataset 분석

<p align="center"><img src="https://github.com/user-attachments/assets/89e046b4-ca4f-4767-ba70-788bed4a6f90" width="90%"></p>

<p align="center"><img src="https://github.com/user-attachments/assets/b4f1a2de-67b7-4f5d-b3fe-6a182543e4c9" width="90%"></p>

### Test Dataset 분석

<p align="center"><img src="https://github.com/user-attachments/assets/962b4328-7edd-4cea-9c57-249dbeacb3ca" width="90%"></p>

<p align="center"><img src="https://github.com/user-attachments/assets/d6e4f6f7-4e6e-4bf9-8ffa-294a1378a769" width="55%"></p>


### Topic 분석

<p align="center"><img src="https://github.com/user-attachments/assets/80a364de-b74f-4a1a-82ee-cffc4557376d" width="45%">
  <img src="https://github.com/user-attachments/assets/65bbc7d2-bafe-496a-b9d5-47bb8ce83321" width="45%"></p>

## 4. 데이터 전처리

### 전처리 과정

주요 공통 전처리 단계:

1. **노이즈 제거**: 특수문자, 반복 문자 정규화
2. **토크나이징**: 모델별 토크나이저를 사용한 토큰화

<div align="center">

  <figure>
    <img src="https://github.com/user-attachments/assets/5ab8936e-9f6f-4f38-90b6-12ecaa04845d" width="90%">
    <br>
    <figcaption>노이즈 제거 규칙</figcaption>
  </figure>

</div>

### 주요 인사이트

- **정보 손실 최소화**: Encoder와 Decoder의 길이 및 토큰 수를 충분히 확보하여 긴 대화의 핵심 정보 보존하는 것이 중요
- **한국어 최적화**: 한국어 요약 태스크에 최적화된 모델 선택 (paust/pko-t5-large 등)  

## 5. 데이터 증강

총 6,000개의 증강 데이터 생성  

### 증강 기법

#### 1. Paraphrase 재작성 (2,500개)

- **목적**: 대화의 의미는 유지하면서 문체와 어휘를 다양화  
- **방법**: Upstage Solar API를 이용해 각 대화(dialogue)와 요약(summary)을 자연스럽게 다시 표현  
- **효과**: 데이터 다양성 향상 → 모델의 일반화 성능 강화  

#### 2. Speaker Swap (2,500개)

- **목적**: 화자 역할을 바꿔 학습 데이터 균형 확보  
- **방법**: `#Person1#` ↔ `#Person2#`의 발화를 상호 교환  
- **효과**: 발화 순서 및 문맥 변화에 대한 모델의 강건성 강화  

#### 3. Synthetic Generation (1,000개)

- **목적**: 새로운 대화 시나리오 생성  
- **방법**: Upstage Solar API를 이용해 주제(topic) 기반의 신규 대화 + 요약 쌍 생성  
- **효과**: 데이터 부족 주제 보완 및 모델 학습 커버리지 확대  

## 6. 파이프라인별 모델링 및 실험 결과

본 프로젝트에서는 두 가지 아키텍처를 실험했습니다:

- Encoder-Decoder Architecture
- Decoder-only Architecture (LLM)

### Pipeline 1: Encoder-Decoder Architecture

**사용 모델**: T5, KoBART, pko-t5-large

#### 주요 전략

1. **길이 설정 최적화**
   - 초기 베이스라인: max_len이 짧아 정보 손실 발생
   - 개선: Encoder/Decoder 길이를 충분히 확보하여 성능 향상

2. **한국어 최적화 모델 전환**
   - 한국어 요약 태스크에 최적화된 `paust/pko-t5-large` 모델 채택
   - 한국어 특성을 반영한 사전학습으로 성능 개선

3. **Topic 정보 활용**
   - **배경**: 대화문만으로는 요약 대상의 주제가 불분명한 경우 발생
   - **방법**: Solar Pro2 API로 Topic 추출 후 입력에 포함

     ```txt
     <topic> {topic} </topic>
     <dialogue> {dialogue} </dialogue>
     ```

- Topic 추가 효과 예시

| Topic | Topic 없이 | Topic과 함께 |
|------|-----------|-------------|
| **공예품 쇼핑** | #Person1#은 중국의 손재주를 보고 싶어합니다. #Person2#는 독특한 중국식으로 운반이 쉬운 종이 공예품, 자수, 바틱을 추천합니다. | #Person2#는 중국의 손재주가 뛰어난 <span style="color:red">공예품</span>을 사려고 합니다. #Person1#은 독특한 중국식이면서 운반이 쉬운 종이 공예품, 자수, 바틱을 추천합니다. |
| **알레르기 식단** | Sue는 Bill의 생일 파티에 케이크를 가져오지 못해 아쉬워한다. Bill은 Sue에게 샐러드를 가져다주고, Sue가 좋아하는 뜨거운 스프를 가져다주겠다고 제안한다. | Sue는 Bill에게 자신의 생일 파티에 케이크를 가져오라고 제안한다. Bill은 <span style="color:red">알레르기 식단 관리</span>를 위해 특정 음식을 피하고 있으며, Sue가 가져온 샌드위치도 먹지 못했다. |

→ Topic 정보를 포함하면 **핵심 맥락과 "왜"** 를 더 정확하게 설명

#### 실험 결과

| Model      | 출력 길이 · 형태 제어 | 탐색 강도 | 반복 억제 · 제약 | Topic 포함여부 | Rouge-1 | Rouge-2 | Rouge-L | Final  |
|-------------|------------------------|------------|------------------|----------------|----------|----------|----------|--------|
| **T5 (best)** | decoder_max_len: 150 <br> encoder_max_len: 1024 | num_beams: 4 | no_repeat_ngram_size: 3 | X | 0.5750 | 0.3766 | 0.4881 | **47.9550** |
| **T5** | length_penalty: 1.2 <br> repetition_penalty: 1.2 | num_beams: 6 | no_repeat_ngram_size: 4 | O | 0.5731 | 0.3642 | 0.4850 | 47.4114 |
| **KoBART** | decoder_max_len: 1024 <br> encoder_max_len: 1024 | num_beams: 6 | no_repeat_ngram_size: 3 | X | 0.5478 | 0.3447 | 0.4578 | 45.0112 |

##### 결과물 예시

<p align="center"><img src="https://github.com/user-attachments/assets/fff480c9-5ae7-4fa1-8f9c-5c520dd55072" width="85%"></p>

---

### Pipeline 2: Decoder-only Architecture (LLM)

**사용 모델**: SOLAR 10.7B/22B, CLOVA X 1.5B, Qwen3 0.6B

#### 주요 전략

1. **Zero-shot / Few-shot / Fine-tuning 비교**
   - LLM의 강력한 언어 이해 능력 활용
   - 학습 없는 추론도 우수한 요약 생성 (Clova X)
   - 단, ROUGE 점수는 상대적으로 낮음

2. **프롬프트 엔지니어링**
   - ROUGE 점수 최적화를 위한 프롬프트 설계
   - Few-shot 예시 포함으로 성능 향상

3. **QLoRA Fine-tuning**
   - 4-bit 양자화로 메모리 효율성 극대화
   - LoRA Adapter만 학습하여 리소스 절약
   - 제한된 GPU 환경(RTX 3090 24GB)에서 대규모 모델 학습 가능

#### 주요 과제 및 해결

문제: Tokenizer 길이 제약

<p align="center"><img src="https://github.com/user-attachments/assets/458c8c7d-885a-4936-8079-60b627ff0a81" width="85%"></p>

- **실험 환경**: LLM은 BART보다 훨씬 많은 토큰 처리 필요
- **환경 제약**: GPU RTX 3090 (VRAM 24GB)
- **대응**:
  - Batch size 1로 설정
  - 가능한 한 큰 max_token 설정 (2048~4096)
  - Gradient accumulation으로 효과적인 배치 크기 확보
- **한계**: 성능 향상은 제한적이었으나, OOM 방지하며 학습 완료

#### 실험 결과

| Model      | Size  | Fine-tuning | Prompt Setting | Rouge-1 | Rouge-2 | Rouge-L | Final  |
|-------------|-------|--------------|----------------|----------|----------|----------|--------|
| **Solar** | 10.7B | QLoRA | One-shot | 0.5500 | 0.3477 | 0.4357 | **44.4480** |
| **Solar Pro** | 22B | QLoRA | Zero-shot | 0.4690 | 0.2910 | 0.3857 | 38.1879 |
| **Qwen3** | 0.6B | QLoRA | Zero-shot | 0.3547 | 0.1261 | 0.2879 | 25.6237 |
| **CLOVA X** | 1.5B | X | Zero-shot | 0.4021 | 0.1712 | 0.3129 | 29.5404 |

##### 결과물 예시

<p align="center"><img src="https://github.com/user-attachments/assets/b45f6d4a-d60a-465c-b4fe-b89b6097afc2" width="85%"></p>

## 7. 리더보드

### 🏆 최종 리더보드 순위
![Rank 1](https://img.shields.io/badge/Leaderboard-Rank%201-gold)
![Final Score 47.9550](https://img.shields.io/badge/Final%20Score-47.9550-blue)

<p align="center"><img src="https://github.com/user-attachments/assets/7ba03161-ec41-48b8-8a57-8304eb437727" width="80%"></p>

### 주요 인사이트

1. **사전학습의 중요성**:
   - 한국어 특화 모델 중에서도 대화문 요약에 적합한 모델을 선택하는 것이 중요함  
   - 이는 현재 수행 중인 과제(Task)와 유사한 데이터로 사전학습(pretraining)된 모델일수록 더 높은 성능을 낼 가능성이 큼을 시사함  

2. **Topic 정보의 효과**:  
   - 주제를 명시적으로 제공하면 요약의 방향성과 초점 명확화  
   - 문맥 이해 향상  

3. **LLM의 가능성과**:  
   - 자연스러운 요약 생성에는 우수  
   - ROUGE 점수 최적화에는 추가 튜닝 필요  
   - QLoRA로 제한된 리소스에서도 대규모 모델 활용 가능  

4. **데이터 증강의 중요성**:  
   - 다양한 증강 기법으로 일반화 성능 향상 (Solar 10.7b)  
   - 특히 Paraphrase와 Speaker Swap이 효과적  

## 8. 실행 가이드

### 공통 환경 설정

```bash
# Python 3.10 환경 생성
conda create -n nlp-summary python=3.10
conda activate nlp-summary

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 파이프라인별 실행

각 파이프라인은 독립적으로 실행됩니다. 상세한 실행 방법은 각 폴더의 README.md를 참고하세요.

## 9. 회고

### 목표 달성도

- **1등 달성**: 최종 ROUGE Final Score 47.9550으로 1위 달성
- **다양한 아키텍처 비교**: Encoder-Decoder vs Decoder-only 성능 비교 완료
- **데이터 증강 효과 검증**: Paraphrase, Speaker Swap, Synthetic 증강의 효과 확인
- **QLoRA 활용**: 제한된 리소스에서 대규모 LLM Fine-tuning 성공
- **Topic 정보 활용**: 주제 기반 요약의 성능 향상 입증

<details>
<summary><b>팀원 회고 보기 👇</b></summary>

| 이름 | 소감 |
|------|------|
| **김장원** | LLM 파인튜닝을 처음 경험하며 거대 언어 모델이 데이터의 뉘앙스와 표현 방식을 어떻게 학습해 내는지 직접 확인할 수 있었습니다. <br> 또한 QLoRA와 같은 효율적인 파인튜닝 기법을 실제 프로젝트에 적용하며 그 원리를 체감하였으며, <br> 결과물의 품질을 결정짓는 프롬프트 설계의 중요성을 깨닫는 계기가 되었습니다. |
| **김영** | 학습과 추론에 시간을 많이 쓰게 되어 시도해보지 못한 부분들에 대한 아쉬움이 살짝 남는 거 같습니다. |
| **문채린** | 직접 실험하고 수정하면서 자연어 처리의 흐름을 익힐 수 있었고 제출 결과값은 아쉬웠지만 이번에도 많이 배웠습니다. |
| **민병호** | 자연어 처리에 대한 어려움과 LLM의 가능성에 대해 다시 한 번 느낄 수 있던 계기가 되었습니다. |
| **이윤서** | 실험 결과가 큰 점수 향상으로 이어지지 않아 아쉬웠지만, 시행착오 속에서도 모델이 점점 똑똑해지는 과정을 보는 게 즐거웠습니다. |
| **정민지** | 초기 베이스라인에서 런타임·버전 이슈가 있었지만, 팀원들의 공유 덕분에 잘 마무리할 수 있었습니다. |

</details>

## 10. 발표 자료

- [프로젝트 발표 슬라이드](https://docs.google.com/presentation/d/1hOjNn1falm06sLyqFpfdF37SkP_zCUD_/)

## 11. 참고 자료

### 논문

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 아키텍처
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### 라이브러리

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## License

본 프로젝트는 Upstage AI Lab 14기 교육 과정의 일환으로 진행되었습니다.
