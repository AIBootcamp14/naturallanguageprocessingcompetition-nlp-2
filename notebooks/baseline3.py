import os
import re
import json
import yaml
from glob import glob
from pprint import pprint
from typing import List, Dict, Tuple, Any, Optional, Union

import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from rouge import Rouge
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
import wandb
from transformers.modeling_utils import PreTrainedModel


# --------------------------------------------------------------------------------------
# 전역 설정 및 PEP 8 준수
# --------------------------------------------------------------------------------------
CONFIG_PATH: str = "./config.yaml"


# --------------------------------------------------------------------------------------
# config.yaml 파일에서 설정값을 직접 불러오도록 변경
# --------------------------------------------------------------------------------------
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        loaded_config: Dict[str, Any] = yaml.safe_load(file)
except FileNotFoundError:
    print(f"오류: 설정 파일 '{CONFIG_PATH}'을 찾을 수 없습니다. config.yaml을 먼저 생성하세요.")
    exit()

# 불러온 config 내용을 출력
pprint(loaded_config)
print(loaded_config.get('general'))
print(loaded_config.get('tokenizer'))
print(loaded_config.get('training'))
print(loaded_config.get('wandb'))
print(loaded_config.get('inference'))

data_path: str = loaded_config['general']['data_path']

# train data와 validation data 불러오기
try:
    train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, 'train.csv'))
    print(train_df.tail())
    val_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, 'dev.csv'))
    print(val_df.tail())
except FileNotFoundError as e:
    print(f"데이터 파일 로드 오류: {e}. 'data_path' 설정을 확인해.")


class Preprocess:
    """
    데이터 전처리를 위한 클래스. 데이터셋을 데이터프레임으로 변환하고
    인코더(dialogue)와 디코더(summary)의 입력을 생성.
    """
    def __init__(
        self,
        bos_token: str,
        eos_token: str,
    ) -> None:
        self.bos_token: str = bos_token
        self.eos_token: str = eos_token

    @staticmethod
    def make_set_as_df(file_path: str, is_train: bool = True) -> pd.DataFrame:
        """실험에 필요한 컬럼(fname, dialogue, summary)을 가진 데이터프레임을 생성."""
        df: pd.DataFrame = pd.read_csv(file_path)
        if is_train:
            # PEP 8: 리턴하는 딕셔너리/리스트에 공백 추가
            return df[['fname', 'dialogue', 'summary', 'topic']]

        return df[['fname', 'dialogue']]

    # PEP 484: Union 대신 ' | ' 사용 (Python 3.10+ 기준)
    def make_input(
        self,
        dataset: pd.DataFrame,
        is_test: bool = False
    ) -> Tuple[List[str], List[str]] | Tuple[List[str], List[str], List[str]]:
        """BART/T5 모델의 입력 형태를 맞추기 위해 전처리를 진행."""
        # \n 및 <br> 같은 줄 바꿈 노이즈를 공백으로 통일
        cleaned_dialogues: pd.Series = dataset['dialogue'].apply(
            lambda x: str(x).replace('\n', ' ').replace('<br>', ' ')
        )

        if is_test:
            encoder_input: List[str] = cleaned_dialogues.tolist()
            # 테스트 시에는 실제 요약 대신 BOS 토큰만 디코더 입력으로 사용
            decoder_input: List[str] = [self.bos_token] * len(cleaned_dialogues)
            return encoder_input, decoder_input
        else:
            encoder_input: List[str] = cleaned_dialogues.tolist()
            # Ground truth를 디코더의 input으로 사용 (BOS 토큰 추가)
            # PEP 8: 줄 바꿈에 주의
            decoder_input: List[str] = dataset['summary'].apply(
                lambda x: self.bos_token + str(x)
            ).tolist()
            # Ground truth를 레이블로 사용 (EOS 토큰 추가)
            # PEP 8: 줄 바꿈에 주의
            decoder_output: List[str] = dataset['summary'].apply(
                lambda x: str(x) + self.eos_token
            ).tolist()
            return encoder_input, decoder_input, decoder_output


class DatasetForTrain(Dataset):
    """모델 학습에 사용되는 Dataset 클래스."""
    def __init__(
        self,
        encoder_input: Dict[str, torch.Tensor],
        decoder_input: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        length: int
    ) -> None:
        self.encoder_input: Dict[str, torch.Tensor] = encoder_input
        self.decoder_input: Dict[str, torch.Tensor] = decoder_input
        self.labels: Dict[str, torch.Tensor] = labels
        self._length: int = length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # PEP 8: 딕셔너리 컴프리헨션 '{key: val[idx] ...}'에서 ':' 뒤에 공백 하나
        item: Dict[str, torch.Tensor] = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item_decoder: Dict[str, torch.Tensor] = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}

        item_decoder['decoder_input_ids'] = item_decoder['input_ids']
        item_decoder['decoder_attention_mask'] = item_decoder['attention_mask']
        item_decoder.pop('input_ids')
        # 'attention_ids' 키는 존재하지 않으므로 제거 시 오류 발생 방지 (pop에 default 값 사용)
        item_decoder.pop('attention_ids', None)
        item_decoder.pop('attention_mask')
        
        item.update(item_decoder)
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        return item

    def __len__(self) -> int:
        return self._length


# Validation에 사용되는 Dataset 클래스는 Train과 동일한 구조를 가집니다.
class DatasetForVal(DatasetForTrain):
    """모델 검증에 사용되는 Dataset 클래스."""
    pass


class DatasetForInference(Dataset):
    """모델 추론에 사용되는 Dataset 클래스."""
    def __init__(self, encoder_input: Dict[str, torch.Tensor], test_id: pd.Series, length: int) -> None:
        self.encoder_input: Dict[str, torch.Tensor] = encoder_input
        self.test_id: pd.Series = test_id
        self._length: int = length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # PEP 8: 딕셔너리 컴프리헨션 '{key: val[idx] ...}'에서 ':' 뒤에 공백 하나
        item: Dict[str, torch.Tensor] = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        # ID는 문자열이므로 텐서가 아닌 그대로 반환
        item['ID'] = self.test_id.iloc[idx]
        return item

    def __len__(self) -> int:
        return self._length


def prepare_train_dataset(
    config: Dict[str, Any],
    preprocessor: Preprocess,
    data_path: str,
    tokenizer: AutoTokenizer
) -> Tuple[DatasetForTrain, DatasetForVal]:
    """훈련 및 검증 데이터셋을 로드하고 토큰화하여 Dataset 객체로 반환."""
    train_file_path: str = os.path.join(data_path, 'train.csv')
    val_file_path: str = os.path.join(data_path, 'dev.csv')

    train_data: pd.DataFrame = preprocessor.make_set_as_df(train_file_path)
    val_data: pd.DataFrame = preprocessor.make_set_as_df(val_file_path)

    print('-' * 150)
    print(f'train_data:\n{train_data["dialogue"].iloc[0]}')
    print(f'train_label:\n{train_data["summary"].iloc[0]}')

    print('-' * 150)
    print(f'val_data:\n{val_data["dialogue"].iloc[0]}')
    print(f'val_label:\n{val_data["summary"].iloc[0]}')

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-' * 10, 'Load data complete', '-' * 10)

    tokenizer_config: Dict[str, Any] = config['tokenizer']

    # 토크나이징 (훈련 데이터)
    tokenized_encoder_inputs: Dict[str, torch.Tensor] = tokenizer(
        encoder_input_train, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['encoder_max_len'],
        return_token_type_ids=False
    )
    tokenized_decoder_inputs: Dict[str, torch.Tensor] = tokenizer(
        decoder_input_train, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['decoder_max_len'],
        return_token_type_ids=False
    )
    tokenized_decoder_ouputs: Dict[str, torch.Tensor] = tokenizer(
        decoder_output_train, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['decoder_max_len'],
        return_token_type_ids=False
    )

    train_inputs_dataset: DatasetForTrain = DatasetForTrain(
        tokenized_encoder_inputs, tokenized_decoder_inputs,
        tokenized_decoder_ouputs, len(encoder_input_train)
    )

    # 토크나이징 (검증 데이터)
    val_tokenized_encoder_inputs: Dict[str, torch.Tensor] = tokenizer(
        encoder_input_val, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['encoder_max_len'],
        return_token_type_ids=False
    )
    val_tokenized_decoder_inputs: Dict[str, torch.Tensor] = tokenizer(
        decoder_input_val, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['decoder_max_len'],
        return_token_type_ids=False
    )
    val_tokenized_decoder_ouputs: Dict[str, torch.Tensor] = tokenizer(
        decoder_output_val, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['decoder_max_len'],
        return_token_type_ids=False
    )

    val_inputs_dataset: DatasetForVal = DatasetForVal(
        val_tokenized_encoder_inputs, val_tokenized_decoder_inputs,
        val_tokenized_decoder_ouputs, len(encoder_input_val)
    )

    print('-' * 10, 'Make dataset complete', '-' * 10)
    return train_inputs_dataset, val_inputs_dataset


def compute_metrics(config: Dict[str, Any], tokenizer: AutoTokenizer, pred: Any) -> Dict[str, float]:
    """
    모델 예측 결과(pred)를 받아 ROUGE 점수를 계산하여 반환.
    """
    try:
        rouge_scorer: Rouge = Rouge()

        predictions: Any = pred.predictions
        labels: Any = pred.label_ids

        # -100 (ignore_index)을 실제 패딩 토큰 ID로 대체
        predictions[predictions == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id

        decoded_preds: List[str] = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
        decoded_labels: List[str] = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

        # 불필요한 생성토큰 제거
        replaced_predictions: List[str] = decoded_preds.copy()
        replaced_labels: List[str] = decoded_labels.copy()
        remove_tokens: List[str] = config['inference']['remove_tokens']

        for token in remove_tokens:
            replaced_predictions = [sentence.replace(token, " ").strip() for sentence in replaced_predictions]
            replaced_labels = [sentence.replace(token, " ").strip() for sentence in replaced_labels]

        # 로그 출력 (첫 3개 샘플)
        print('-' * 150)
        print(f"PRED: {replaced_predictions[0]}")
        print(f"GOLD: {replaced_labels[0]}")
        print('-' * 150)
        print(f"PRED: {replaced_predictions[1]}")
        print(f"GOLD: {replaced_labels[1]}")
        print('-' * 150)
        print(f"PRED: {replaced_predictions[2]}")
        print(f"GOLD: {replaced_labels[2]}")
        print('-' * 150)

        # 최종적인 ROUGE 점수를 계산
        # ROUGE 라이브러리는 빈 문자열에서 오류가 날 수 있으므로 빈 문자열은 제거
        valid_preds: List[str] = [p for p in replaced_predictions if p.strip()]
        valid_labels: List[str] = [l for p, l in zip(replaced_predictions, replaced_labels) if p.strip()]

        if not valid_preds:
            print("경고: 유효한 예측 문자열이 없습니다.")
            return {}

        # get_scores의 입력으로 빈 리스트가 들어가지 않도록 유효성 검사
        if not valid_preds or not valid_labels:
            print("경고: 유효한 예측 또는 레이블이 없어 ROUGE 점수를 계산할 수 없습니다.")
            return {}

        results: Dict[str, Dict[str, float]] = rouge_scorer.get_scores(valid_preds, valid_labels, avg=True)

        # ROUGE 점수 중 F-1 score를 통해 평가
        result: Dict[str, float] = {key: value["f"] for key, value in results.items()}

        # 소수점 4자리까지 반올림
        result = {k: round(v * 100, 4) for k, v in result.items()}

        return result

    except Exception as e:
        print(f"Error computing metrics: {e}")
        # 오류 발생 시 빈 딕셔너리를 반환하여 Trainer가 멈추지 않도록 함
        return {}


def load_trainer_for_train(
    config: Dict[str, Any],
    generate_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    train_inputs_dataset: DatasetForTrain,
    val_inputs_dataset: DatasetForVal
) -> Seq2SeqTrainer:
    """Seq2SeqTrainer 객체를 초기화하고 반환."""
    print('-' * 10, 'Make training arguments', '-' * 10)

    # config의 값을 직접 사용하여 Seq2SeqTrainingArguments 생성
    training_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to'],
        # 메모리 절약 기능 활성화 (config.yaml에서 값을 가져옴)
        gradient_checkpointing=config['training'].get('gradient_checkpointing', False),
        # PEP 8: eval_strategy로 변경 (버전업에 대비)
        eval_strategy=config['training']['evaluation_strategy'], 
    )

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능
    my_callback: EarlyStoppingCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    print('-' * 10, 'Make training arguments complete', '-' * 10)
    print('-' * 10, 'Make trainer', '-' * 10)

    # Trainer 클래스 정의
    trainer: Seq2SeqTrainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[my_callback]
    )
    print('-' * 10, 'Make trainer complete', '-' * 10)

    return trainer


def load_tokenizer_and_model_for_train(
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """학습을 위한 tokenizer와 사전 학습된 모델을 불러옴."""
    print('-' * 10, 'Load tokenizer & model', '-' * 10)

    model_name: str = config['general']['model_name']
    print('-' * 10, f'Model Name : {model_name}', '-' * 10)

    # config에 설정된 모델 이름으로 토크나이저와 모델을 로드
    local_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    special_tokens_dict: Dict[str, List[str]] = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    # 안전하게 다시 추가하고 resize_token_embeddings 호출
    local_tokenizer.add_special_tokens(special_tokens_dict)

    # special token을 추가했으므로 임베딩 레이어 크기 재조정
    generate_model.resize_token_embeddings(len(local_tokenizer))
    generate_model.to(device)
    print(generate_model.config)

    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)
    return generate_model, local_tokenizer


def find_latest_checkpoint(output_dir: str = './') -> Optional[str]:
    """가장 최근에 저장된 체크포인트 폴더 경로를 찾는다."""
    # glob 모듈을 사용하여 'checkpoint-'로 시작하는 모든 폴더를 찾음
    checkpoint_dirs: List[str] = glob(f"{output_dir}/checkpoint-*")
    
    if not checkpoint_dirs:
        return None
        
    # 폴더 이름 대신, 수정 시간을 기준으로 가장 최근의 폴더를 선택
    # os.path.getmtime: 파일/폴더의 마지막 수정 시간을 초 단위로 반환
    latest_checkpoint: str = max(checkpoint_dirs, key=os.path.getmtime)
    
    return latest_checkpoint


def update_inference_path_with_latest_checkpoint(config: Dict[str, Any]) -> None:
    """학습 폴더에서 가장 최근 체크포인트 경로를 찾아 config를 업데이트한다."""
    output_dir: str = config['general']['output_dir']
    
    # load_best_model_at_end가 True일 때, 가장 최적의 모델이 마지막 체크포인트 폴더에 저장됨
    latest_path: Optional[str] = find_latest_checkpoint(output_dir)

    if latest_path:
        # 찾은 최신 경로로 config['inference']['ckt_path']를 덮어쓰기
        config['inference']['ckt_path'] = latest_path 
        print(f"\n✨ [자동 설정] 최신 체크포인트 경로: '{latest_path}'로 업데이트되었습니다.")
        # 업데이트된 경로를 저장 (선택 사항이지만 안전을 위해)
        try:
            # PEP 8: 파일 모드 인수에 공백 제거
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
        except Exception as e:
            print(f"경고: config.yaml 파일 업데이트 중 오류 발생 - {e}")
    else:
        print("\n⚠️ [자동 설정 실패] 체크포인트 폴더를 찾을 수 없습니다. config.yaml의 ckt_path를 사용합니다.")
    

def prepare_test_dataset(
    config: Dict[str, Any],
    preprocessor: Preprocess,
    tokenizer: AutoTokenizer
) -> Tuple[pd.DataFrame, DatasetForInference]:
    """테스트 데이터셋을 로드하고 토큰화하여 Dataset 객체로 반환."""
    test_file_path: str = os.path.join(config['general']['data_path'], 'test.csv')

    test_data: pd.DataFrame = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id: pd.Series = test_data['fname']

    print('-' * 150)
    print(f'test_data:\n{test_data["dialogue"].iloc[0]}')
    print('-' * 150)

    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('-' * 10, 'Load data complete', '-' * 10)

    tokenizer_config: Dict[str, Any] = config['tokenizer']

    # 토크나이징 (인코더 입력)
    test_tokenized_encoder_inputs: Dict[str, torch.Tensor] = tokenizer(
        encoder_input_test, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['encoder_max_len'],
        return_token_type_ids=False
    )
    # 토크나이징 (디코더 입력 - T5에서는 필요 없지만 구조 유지를 위해 포함)
    _test_tokenized_decoder_inputs: Dict[str, torch.Tensor] = tokenizer(
        decoder_input_test, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True,
        max_length=tokenizer_config['decoder_max_len'],
        return_token_type_ids=False
    )

    test_encoder_inputs_dataset: DatasetForInference = DatasetForInference(
        test_tokenized_encoder_inputs, test_id, len(encoder_input_test)
    )
    print('-' * 10, 'Make dataset complete', '-' * 10)

    return test_data, test_encoder_inputs_dataset


def load_tokenizer_and_model_for_test(
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """추론을 위한 tokenizer와 학습된 모델을 불러옴."""
    print('-' * 10, 'Load tokenizer & model', '-' * 10)

    model_name: str = config['general']['model_name']
    ckt_path: str = config['inference']['ckt_path']
    print('-' * 10, f'Model Name : {model_name}', '-' * 10)
    print('-' * 10, f'Checkpoint Path : {ckt_path}', '-' * 10)

    # config에 설정된 모델 이름으로 토크나이저를 로드
    local_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict: Dict[str, List[str]] = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    local_tokenizer.add_special_tokens(special_tokens_dict)

    # 💥💥💥 [핵심 수정] safe_serialization=False 옵션을 제거함! 💥💥💥
    generate_model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
        ckt_path, 
        low_cpu_mem_usage=True
    )
    generate_model.resize_token_embeddings(len(local_tokenizer))
    generate_model.to(device)
    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)

    return generate_model, local_tokenizer


def inference(config: Dict[str, Any]) -> pd.DataFrame:
    """학습된 모델을 사용하여 추론을 수행하고 결과를 DataFrame으로 반환."""
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 10, f'device : {device}', '-' * 10)
    print(torch.__version__)

    generate_model: AutoModelForSeq2SeqLM
    local_tokenizer: AutoTokenizer
    try:
        generate_model, local_tokenizer = load_tokenizer_and_model_for_test(config, device)
    except Exception as e:
        print(f"모델 로드 오류: {e}. 'ckt_path'를 확인하세요.")
        return pd.DataFrame() 

    # T5 모델의 BOS/EOS 토큰은 토크나이저의 기본값을 사용하도록 수정
    bos_token: str = local_tokenizer.bos_token if local_tokenizer.bos_token else local_tokenizer.pad_token
    eos_token: str = local_tokenizer.eos_token if local_tokenizer.eos_token else "</s>"

    preprocessor: Preprocess = Preprocess(bos_token, eos_token)

    test_data: pd.DataFrame
    test_encoder_inputs_dataset: DatasetForInference
    try:
        test_data, test_encoder_inputs_dataset = prepare_test_dataset(
            config, preprocessor, local_tokenizer)
    except Exception as e:
        print(f"테스트 데이터셋 준비 오류: {e}. 'data_path'를 확인하세요.")
        return pd.DataFrame() 

    dataloader: DataLoader = DataLoader(
        test_encoder_inputs_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )

    summary: List[str] = []
    text_ids: List[str] = []
    generate_model.eval() 
    with torch.no_grad():
        for item in tqdm(dataloader):
            # ID는 문자열(str)이므로 리스트에 담을 때 extend
            text_ids.extend(item['ID'])

            # CUDA 장치로 텐서를 이동
            # PEP 8: 딕셔너리 접근 시 공백 제거
            input_ids: torch.Tensor = item['input_ids'].to(device)

            generated_ids: torch.Tensor = generate_model.generate(
                input_ids=input_ids,
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )
            for ids in generated_ids:
                result: str = local_tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거
    remove_tokens: List[str] = config['inference']['remove_tokens']
    preprocessed_summary: List[str] = summary.copy()
    for token in remove_tokens:
        # replace 후 strip을 추가하여 깔끔한 결과 보장
        preprocessed_summary = [sentence.replace(token, " ").strip() for sentence in preprocessed_summary]

    output: pd.DataFrame = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary": preprocessed_summary,
        }
    )
    result_path: str = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output


def main(config: Dict[str, Any]) -> None:
    """
    학습 또는 추론을 시작하는 메인 함수.
    config['training']['do_train'] 값에 따라 작동이 달라짐.
    """
    if config['training']['do_train']:
        print("\n=== 모델 학습 시작 (do_train: True) ===")
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('-' * 10, f'device : {device}', '-' * 10)

        # 1. 토크나이저 및 모델 로드
        generate_model: AutoModelForSeq2SeqLM
        local_tokenizer: AutoTokenizer
        generate_model, local_tokenizer = load_tokenizer_and_model_for_train(config, device)

        # 2. 데이터 전처리기 준비
        bos_token: str = local_tokenizer.bos_token if local_tokenizer.bos_token else local_tokenizer.pad_token
        eos_token: str = local_tokenizer.eos_token if local_tokenizer.eos_token else "</s>"
        preprocessor: Preprocess = Preprocess(bos_token, eos_token)

        # 3. 데이터셋 준비
        data_path: str = config['general']['data_path']
        train_inputs_dataset: DatasetForTrain
        val_inputs_dataset: DatasetForVal
        train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(
            config, preprocessor, data_path, local_tokenizer
        )

        # 4. Trainer 로드 및 학습 시작
        trainer: Seq2SeqTrainer = load_trainer_for_train(
            config, generate_model, local_tokenizer, train_inputs_dataset, val_inputs_dataset
        )
        trainer.train()
        print("\n=== 모델 학습 완료 ===")
    else:
        print("\n=== 추론 모드 (do_train: False) ===\n")
        # 추론 로직은 __main__ 블록에서 별도로 처리


if __name__ == "__main__":
    # training/do_train: True인 경우 main 함수 내에서 학습이 진행됨
    if loaded_config['training']['do_train']:
        main(loaded_config)
    
    # 추론 전 최신 체크포인트를 자동으로 로드하도록 config를 업데이트 (학습을 했을 경우에만 의미 있음)
    update_inference_path_with_latest_checkpoint(loaded_config)

    # config.yaml의 ckt_path를 사용하지 않고, 수동으로 './checkpoint-7008'을 강제 지정
    # 이 부분은 네가 직접 넣어준 강제 설정이므로 그대로 유지함
    loaded_config['inference']['ckt_path'] = './checkpoint-7008'
    print(f"\n✅ [최종 강제 설정] 추론 체크포인트: '{loaded_config['inference']['ckt_path']}'")

    # 추론 실행 (do_train이 False일 때만 실질적인 의미가 있음)
    output: pd.DataFrame = inference(loaded_config)

    print(output)