import os, torch, argparse
import pandas as pd

from src.config import load_config
from src.inference.inference import inference
from src.utils.gpu_utils import clean_gpu_memory
from src.model.model_utils import load_tokenizer_and_model_for_train
from src.data.preprocess.preprocess import Preprocess
from src.data.dataset import DatasetForTrain, DatasetForVal
from src.train.train import load_trainer_for_train, compute_metrics
from src.llm.exaone import load_and_infer_exaone, load_and_train_exaone
from src.data.dataset import DatasetForInference
from src.data.preprocess.preprocess import prepare_test_dataset, prepare_train_dataset

CONFIG_PATH = "../configs/config.yaml"


def run_train(config):
    
    clean_gpu_memory()
    
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    
    # -------------자동으로 최신 체크포인트 찾기----------------------
    # resume_from = find_latest_checkpoint(config['general']['output_dir'])
    
    # if resume_from:
    #     print(f"🔄 체크포인트 발견! 재개합니다: {resume_from}")
    #     user_input = input("계속하시겠습니까? (y/n): ")
    #     if user_input.lower() != 'y':
    #         resume_from = None
    # -------------자동으로 최신 체크포인트 찾기----------------------        
            
    generate_model , tokenizer = load_tokenizer_and_model_for_train(config,device)
    
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token']) 
    
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config,preprocessor, data_path, tokenizer)
    
    trainer = load_trainer_for_train(config, generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset)
    trainer.train() 
    
    # 모델 및 토크나이저 저장
    trainer.save_model(config['general']['output_dir']+"/best_model")   
    tokenizer.save_pretrained(config['general']['output_dir']+"/best_model")


    
    
def run_infer(config):
    
    clean_gpu_memory()
    
    # ckt_path ="/workspace/NLP_Dialogue_Summarization/output/model/best_model"
    ckt_path ="/workspace/NLP_Dialogue_Summarization/output/model/3.best_model_training_config"
    # ckt_name = "checkpoint-6200"
    # config['inference']['ckt_path'] = ckt_path + ckt_name
    config['inference']['ckt_path'] = ckt_path
    
    return inference(config)


def run_train_exaone(config):
    clean_gpu_memory()
    load_and_train_exaone()
    

def run_infer_exaone(config):
    clean_gpu_memory()
    load_and_infer_exaone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=["train", "infer", "exaone"], help='train or infer')
    args = parser.parse_args()
    
    config = load_config(CONFIG_PATH)
    mode = args.mode
    
    if mode == "train":
        # run_train(config)
        run_train_exaone(config)
    elif mode == "exaone":
        run_infer_exaone(config)
    else:
        run_infer(config)    
        
        
def find_invalid_topics(df, max_topic_len: int = 30, min_confidence: float = 0.7):
    """
    길이가 너무 긴 topic 또는 confidence가 낮은 데이터를 찾아냄.

    Args:
        df (pd.DataFrame): fname, dialogue, topic, topic_confidence 컬럼을 포함한 데이터프레임
        max_topic_len (int): topic 글자 수 제한 기준 (기본 15자 초과 시 경고)
        min_confidence (float): 최소 confidence 기준 (기본 0.7 미만 시 경고)

    Returns:
        pd.DataFrame: 조건에 해당하는 행만 반환
    """
    # 문자열로 안전 변환
    df['topic'] = df['topic'].astype(str)
    df['topic_confidence'] = df['topic_confidence'].astype(float)

    # 조건 설정
    too_long = df['topic'].apply(lambda x: len(x) > max_topic_len)
    low_conf = df['topic_confidence'] < min_confidence

    # 필터링
    invalid_df = df[too_long | low_conf].copy()

    print(f"⚠️ 총 {len(invalid_df)}개의 비정상 topic 탐지됨")
    print(f" - 길이 초과: {(too_long).sum()}개")
    print(f" - confidence < {min_confidence}: {(low_conf).sum()}개\n")

    return invalid_df 


if __name__ == "__main__":
    main()
    # clean_gpu_memory()
    
    # =============== 제출 파일 검증! =================
    df = pd.read_csv("/workspace/NLP_Dialogue_Summarization/data/test_topic_solar.csv")
    invalid_topics = find_invalid_topics(df, max_topic_len=20, min_confidence=0.7)
    print(invalid_topics[['fname', 'topic', 'topic_confidence']].head())
    # =============== 제출 파일 검증! =================