from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,DataCollatorForSeq2Seq
import torch
from rouge import Rouge
from src.data.dataset import DatasetForTrain, DatasetForVal
import os

# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
# def compute_metrics(config,tokenizer,pred):
#     rouge = Rouge()
#     predictions = pred.predictions
#     labels = pred.label_ids

#     predictions[predictions == -100] = tokenizer.pad_token_id
#     labels[labels == -100] = tokenizer.pad_token_id

#     decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
#     labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

#     # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
#     replaced_predictions = decoded_preds.copy()
#     replaced_labels = labels.copy()
#     remove_tokens = config['inference']['remove_tokens']
#     for token in remove_tokens:
#         replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
#         replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

#     print('-'*150)
#     print(f"PRED: {replaced_predictions[0]}")
#     print(f"GOLD: {replaced_labels[0]}")
#     print('-'*150)
#     print(f"PRED: {replaced_predictions[1]}")
#     print(f"GOLD: {replaced_labels[1]}")
#     print('-'*150)
#     print(f"PRED: {replaced_predictions[2]}")
#     print(f"GOLD: {replaced_labels[2]}")

#     # 최종적인 ROUGE 점수를 계산합니다.
#     results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

#     # ROUGE 점수 중 F-1 score를 통해 평가합니다.
#     result = {key: value["f"] for key, value in results.items()}
#     return result


# def compute_metrics(config, tokenizer, pred):
#     """ROUGE 점수 계산 - 빈 예측 처리 강화  (psyche/KoT5-summarization 모델용)"""
    
#     rouge = Rouge()
#     predictions = pred.predictions
#     labels = pred.label_ids

#     # -100을 pad_token_id로 변경
#     predictions[predictions == -100] = tokenizer.pad_token_id
#     labels[labels == -100] = tokenizer.pad_token_id

#     # 디코딩
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # 불필요한 토큰 제거
#     replaced_predictions = decoded_preds.copy()
#     replaced_labels = decoded_labels.copy()
#     remove_tokens = config['inference']['remove_tokens']
    
#     for token in remove_tokens:
#         replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
#         replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]
    
#     # 공백 정리
#     replaced_predictions = [" ".join(pred.split()) for pred in replaced_predictions]
#     replaced_labels = [" ".join(label.split()) for label in replaced_labels]

#     # *** 샘플 출력 (디버깅용) ***
#     print('-'*150)
#     print(f"PRED 0: {replaced_predictions[0][:200]}")
#     print(f"GOLD 0: {replaced_labels[0][:200]}")
#     print('-'*150)
#     if len(replaced_predictions) > 1:
#         print(f"PRED 1: {replaced_predictions[1][:200]}")
#         print(f"GOLD 1: {replaced_labels[1][:200]}")
#         print('-'*150)
#     if len(replaced_predictions) > 2:
#         print(f"PRED 2: {replaced_predictions[2][:200]}")
#         print(f"GOLD 2: {replaced_labels[2][:200]}")
#         print('-'*150)

#     # *** 빈 문자열 처리 (ROUGE 에러 방지) ***
#     # Rouge 라이브러리는 빈 문자열을 처리하지 못함
#     for i in range(len(replaced_predictions)):
#         if not replaced_predictions[i].strip():
#             replaced_predictions[i] = "."
#         if not replaced_labels[i].strip():
#             replaced_labels[i] = "."

#     # ROUGE 점수 계산
#     try:
#         results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
#         result = {key: value["f"] for key, value in results.items()}
#     except Exception as e:
#         print(f"ROUGE 계산 에러: {e}")
#         result = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
#     return result


def compute_metrics(config, tokenizer, pred):
    """ROUGE 점수 계산 - 빈 예측 처리 강화  (lcw99/t5-base-korean-text-summary 모델용)"""
    from rouge import Rouge
    
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    # -100을 pad_token_id로 변경
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 불필요한 토큰 제거
    replaced_predictions = decoded_preds.copy()
    replaced_labels = decoded_labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]
    
    # 공백 정리 및 빈 문자열 처리
    replaced_predictions = [" ".join(pred.strip().split()) for pred in replaced_predictions]
    replaced_labels = [" ".join(label.strip().split()) for label in replaced_labels]

    # *** 핵심: 빈 문자열과 점만 있는 경우 처리 ***
    for i in range(len(replaced_predictions)):
        pred_clean = replaced_predictions[i].replace(".", "").strip()
        if not pred_clean or len(pred_clean) < 3:  # 빈 문자열 또는 너무 짧은 경우
            replaced_predictions[i] = "없음"  # 의미 있는 대체 텍스트
        if not replaced_labels[i].strip():
            replaced_labels[i] = "없음"

    # 샘플 출력
    print('-'*150)
    for idx in range(min(3, len(replaced_predictions))):
        print(f"PRED {idx}: {replaced_predictions[idx][:200]}")
        print(f"GOLD {idx}: {replaced_labels[idx][:200]}")
        print('-'*150)

    # ROUGE 계산
    try:
        results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
        result = {key: value["f"] for key, value in results.items()}
    except Exception as e:
        print(f"ROUGE 계산 에러: {e}")
        print(f"문제가 된 predictions 샘플: {replaced_predictions[:3]}")
        result = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    return result



# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)
    # set training args
    training_args = Seq2SeqTrainingArguments(
                output_dir=config['general']['output_dir'], # model output directory
                overwrite_output_dir=config['training']['overwrite_output_dir'],
                num_train_epochs=config['training']['num_train_epochs'],  # total number of training epochs
                learning_rate=config['training']['learning_rate'], # learning_rate
                per_device_train_batch_size=config['training']['per_device_train_batch_size'], # batch size per device during training
                per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],# batch size for evaluation
                warmup_ratio=config['training']['warmup_ratio'],  # number of warmup steps for learning rate scheduler
                weight_decay=config['training']['weight_decay'],  # strength of weight decay
                lr_scheduler_type=config['training']['lr_scheduler_type'],
                optim =config['training']['optim'],
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                eval_strategy=config['training']['evaluation_strategy'], # evaluation strategy to adopt during training
                save_strategy =config['training']['save_strategy'],
                save_total_limit=config['training']['save_total_limit'], # number of total save model.
                fp16=config['training']['fp16'],
                load_best_model_at_end=config['training']['load_best_model_at_end'], # 최종적으로 가장 높은 점수 저장
                seed=config['training']['seed'],
                logging_dir=config['training']['logging_dir'], # directory for storing logs
                logging_strategy=config['training']['logging_strategy'],
                predict_with_generate=config['training']['predict_with_generate'], #To use BLEU or ROUGE score
                do_train=config['training']['do_train'],
                do_eval=config['training']['do_eval'],
                metric_for_best_model="rouge-l",
                greater_is_better=True,  
                report_to=[],
                generation_max_length=config['training']['generation_max_length'],
                generation_num_beams=6,
                # report_to=config['training']['report_to'], # (선택) wandb를 사용할 때 설정합니다.
                label_smoothing_factor=config['training']['label_smoothing_factor'],  # ← 추가!
                gradient_checkpointing=config['training']['gradient_checkpointing'],  # ← 추가!
                bf16=config['training']['bf16'],  # ← 추가! (fp16이 있는데 bf16이 없네요)
            )

    # (선택) 모델의 학습 과정을 추적하는 wandb를 사용하기 위해 초기화 해줍니다.
    # wandb.init(
    #     entity=config['wandb']['entity'],
    #     project=config['wandb']['project'],
    #     name=config['wandb']['name'],
    #     settings=wandb.Settings(start_method="thread", init_timeout=300)
    # )

    # # (선택) 모델 checkpoint를 wandb에 저장하도록 환경 변수를 설정합니다.
    # os.environ["WANDB_LOG_MODEL"]="false"
    # os.environ["WANDB_WATCH"]="false"

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용합니다.
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=generate_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if (config['training']['bf16'] or config['training']['fp16']) else None,
    )

    # Trainer 클래스를 정의합니다.
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        data_collator=data_collator,
        compute_metrics = lambda pred: compute_metrics(config,tokenizer, pred),
        callbacks = [MyCallback]
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer