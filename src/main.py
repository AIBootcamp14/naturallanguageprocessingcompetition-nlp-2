import os, torch, argparse
import pandas as pd

from src.config import load_config
from src.inference.inference import inference
from src.utils.gpu_utils import clean_gpu_memory
from src.model.model_utils import load_tokenizer_and_model_for_train
from src.data.preprocess.preprocess import Preprocess
from src.data.dataset import DatasetForTrain, DatasetForVal
from src.train.train import load_trainer_for_train, compute_metrics
from src.data.dataset import DatasetForInference
from src.data.preprocess.preprocess import prepare_test_dataset, prepare_train_dataset

CONFIG_PATH = "/workspace/NLP_Dialogue_Summarization/src/config.yaml"

def run_train(config):
    
    clean_gpu_memory()
    
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
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
    ckt_path ="/workspace/NLP_Dialogue_Summarization/output/model/best_model"
    # ckt_name = "checkpoint-6200"
    # config['inference']['ckt_path'] = ckt_path + ckt_name
    config['inference']['ckt_path'] = ckt_path
    
    return inference(config)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=["train", "infer"], help='train or infer')
    args = parser.parse_args()
    
    config = load_config(CONFIG_PATH)
    mode = args.mode
    
    if mode == "train":
        run_train(config)
    else:
        run_infer(config)    



if __name__ == "__main__":
    main()
    # clean_gpu_memory()
    