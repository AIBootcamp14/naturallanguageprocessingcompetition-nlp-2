import pandas as pd
import torch
from tqdm import tqdm
import os
import re
from torch.utils.data import DataLoader
from src.data.preprocess.preprocess import Preprocess
from src.model.model_utils import load_tokenizer_and_model_for_test
from src.data.dataset import DatasetForInference
from src.data.preprocess.preprocess import prepare_test_dataset, prepare_train_dataset

# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to('cuda:0'),
                            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                            early_stopping=config['inference']['early_stopping'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                            
                            temperature=1.0,  # 0.8~1.0 사이 권장
                            do_sample=False,  # beam search 사용 시 False
                            repetition_penalty=2.5,
                            
                            num_beam_groups=config['inference'].get('num_beam_groups', 1),  # 기본값 1
                            diversity_penalty=config['inference'].get('diversity_penalty', 0.0),  # 기본값 0
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]
        
    # ✅ 공백/특수문자 추가 정리
    cleaned_summary = []
    for s in preprocessed_summary:
        s = s.strip()                         # 앞뒤 공백 제거
        s = re.sub(r'\s+', ' ', s)            # 여러 공백 → 하나
        s = s.strip('"').strip("'")           # 따옴표 제거
        cleaned_summary.append(s)    

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : cleaned_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output