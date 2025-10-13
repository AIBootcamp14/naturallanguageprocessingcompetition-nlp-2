import pandas as pd
import os
from tqdm import tqdm
from src.data.dataset import DatasetForTrain, DatasetForVal
from src.data.dataset import DatasetForInference

# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    def __init__(self,
            bos_token: str,
            eos_token: str,
        ) -> None:

        self.bos_token = bos_token
        self.eos_token = eos_token
    
    @staticmethod    
    def preprocess_data(df):
        df = df.copy()
        # fname 정리
        df['fname'] = df['fname'].str.strip()
        
        # topic (없을 수 있으므로 안전 처리)
        if 'topic' in df.columns:
            df['topic'] = df['topic'].astype(str).fillna('unknown')
            df['topic'] = df['topic'].str.replace(r'\s+', ' ', regex=True).str.strip('"').str.strip("'").str.strip()
        else:
            df['topic'] = 'unknown'
        
        # summary 정리
        if 'summary' in df.columns:
            df['summary'] = df['summary'].str.strip()
            df['summary'] = df['summary'].str.replace(r'\s+', ' ', regex=True)
            df['summary'] = df['summary'].str.strip('"').str.strip("'")
        
        # dialogue 정리
        if 'dialogue' in df.columns:
            df['dialogue'] = df['dialogue'].str.strip()
            df['dialogue'] = df['dialogue'].str.replace(r'\s+', ' ', regex=True)
            df['dialogue'] = df['dialogue'].str.strip('"').str.strip("'")
        
        return df
                

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    def make_set_as_df(file_path, is_train = True):
        df = pd.read_csv(file_path)
        # topic 포함하여 읽기 (테스트에도 topic 있을 수 있음)
        needed_cols = ['fname', 'dialogue']
        if 'summary' in df.columns and is_train:
            needed_cols.append('summary')
        if 'topic' in df.columns:
            needed_cols.append('topic')

        df = df[needed_cols]
        df = Preprocess.preprocess_data(df)
        return df
        

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset,is_test = False):
        encoder_input = (
            "<topic> " + dataset['topic'].astype(str) + " </topic>\n"
            + "<dialogue> " + dataset['dialogue'].astype(str) + " </dialogue>"
        )
        print("topic 들어갔는지 여부 샘플 : "+encoder_input[0])
        
        if is_test:
            decoder_input = [self.bos_token] * len(dataset)
            return encoder_input.tolist(), list(decoder_input)
        else:
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
        
        # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input_t5(self, dataset,is_test = False):
        """T5 모델용 입력 생성 - 간소화 버전"""
        # Encoder input: "summarize: " prefix + dialogue
        # encoder_input = (
        #     "summarize: "
        #     + dataset['dialogue'].astype(str)
        # )
        
        # Topic 정보는 dialogue 앞에 자연스럽게 삽입
        # encoder_input = (
        #     "summarize: 주제: " + dataset['topic'].astype(str) + " "
        #     + dataset['dialogue'].astype(str)
        # )
        encoder_input = (
        "summarize: <topic>" + dataset['topic'].astype(str) + "</topic> "
        + dataset['dialogue'].astype(str)
        )
        
        print("="*80)
        print("Encoder Input 샘플:")
        print(encoder_input.iloc[0][:300])
        print("="*80)
        
        if is_test:
            return encoder_input.tolist(), None
        else:
            # Labels: 요약문 (EOS 토큰 제거 - T5가 자동 추가)
            labels = dataset['summary'].astype(str)
            
            print("Labels 샘플:")
            print(labels.iloc[0])
            print("="*80)
            
            return encoder_input.tolist(), labels.tolist()


# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    """T5용 데이터셋 준비 - 간소화 버전"""
    
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    # 데이터 로드
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)
    
    print(f"Train 데이터 크기: {len(train_data)}")
    print(f"Validation 데이터 크기: {len(val_data)}")

    # T5 입력 생성 (decoder_input 제거)
    encoder_input_train, labels_train = preprocessor.make_input_t5(train_data)
    encoder_input_val, labels_val = preprocessor.make_input_t5(val_data)
    
    print('-'*10, 'Load data complete', '-'*10)

    # Tokenization
    tokenized_encoder_train = tokenizer(
        encoder_input_train,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len']
    )
    
    tokenized_labels_train = tokenizer(
        labels_train,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len']
    )
    
    # Validation tokenization
    tokenized_encoder_val = tokenizer(
        encoder_input_val,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len']
    )
    
    tokenized_labels_val = tokenizer(
        labels_val,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len']
    )

    # Dataset 생성 (동일한 클래스 사용)
    train_dataset = DatasetForTrain(
        tokenized_encoder_train,
        tokenized_labels_train,
        len(encoder_input_train),
        tokenizer.pad_token_id  # ← 추가!
    )
    
    val_dataset = DatasetForTrain(
        tokenized_encoder_val,
        tokenized_labels_val,
        len(encoder_input_val),
        tokenizer.pad_token_id  # ← 추가!
    )

    print('-'*10, 'Make dataset complete', '-'*10)
    
    # 샘플 확인
    sample = train_dataset[0]
    print("\nDataset 샘플 확인:")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
    # ✅ -100 포함 여부 (토치 방식)
    has_neg100 = (sample['labels'] == -100).any().item()
    print(f"Labels에 -100 포함 여부: {has_neg100}")

    # ✅ -100이 아닌 값 개수
    num_non_ignored = (sample['labels'] != -100).sum().item()
    print(f"Labels 중 -100이 아닌 값 개수: {num_non_ignored}")
    
    return train_dataset, val_dataset




# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
# def prepare_test_dataset(config,preprocessor, tokenizer):

#     # test_file_path = os.path.join(config['general']['data_path'],'test.csv')
#     # test_file_path = os.path.join(config['general']['data_path'],'test_topic.csv')
#     test_file_path = os.path.join(config['general']['data_path'],'test_topic_solar.csv')

#     test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
#     test_id = test_data['fname']

#     # print('-'*150)
#     # print(f'test_data:\n{test_data["dialogue"][0]}')
#     # print('-'*150)

#     # encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
#     encoder_input_test , decoder_input_test = preprocessor.make_input_t5(test_data,is_test=True)
#     print('-'*10, 'Load data complete', '-'*10,)

#     test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
#                     add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
#     test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
#                     add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

#     test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
#     print('-'*10, 'Make dataset complete', '-'*10,)

#     return test_data, test_encoder_inputs_dataset


def prepare_test_dataset(config, preprocessor, tokenizer):
    """T5용 테스트 데이터셋 준비 - train 스타일에 맞춘 간소화 버전"""
    
    # 테스트 경로 (현재 사용 중인 파일 유지)
    test_file_path = os.path.join(config['general']['data_path'], 'test_topic_solar.csv')

    # 데이터 로드
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']
    print(f"Test 데이터 크기: {len(test_data)}")

    # T5 입력 생성: 테스트는 인코더 입력만 사용 (디코더 입력/라벨 불필요)
    # preprocessor가 (encoder, decoder) 튜플을 반환한다면 decoder는 무시
    encoder_input_test, _ = preprocessor.make_input_t5(test_data, is_test=True)

    print('-'*10, 'Load data complete', '-'*10)

    # Tokenization (인코더 입력만)
    tokenized_encoder_test = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len']
    )

    # Inference용 Dataset 생성 (train과 동일한 텐서 형태 유지)
    test_encoder_inputs_dataset = DatasetForInference(
        tokenized_encoder_test,
        test_id,
        len(encoder_input_test)
    )

    print('-'*10, 'Make dataset complete', '-'*10)

    # 샘플 확인
    sample = test_encoder_inputs_dataset[0]
    print("\n[TEST Dataset 샘플 확인]")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    if 'attention_mask' in sample:
        print(f"attention_mask shape: {sample['attention_mask'].shape}")

    return test_data, test_encoder_inputs_dataset
