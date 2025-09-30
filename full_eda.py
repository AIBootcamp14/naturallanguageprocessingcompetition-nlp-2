import pandas as pd
from transformers import AutoTokenizer
import os
import numpy as np # Numpy를 추가로 사용해서 안정적인 계산을 할 거야
from pprint import pprint
from typing import Dict, Any

# 1. Config와 Tokenizer 로드 (이전에 사용된 것과 동일)
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")

# 2. 데이터 로드 경로 설정 (사용자 경로와 일치하도록 수정: ./data/)
DATA_PATH = "./data/" 

def calculate_token_lengths(text_series: pd.Series, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """주어진 텍스트 Series의 토큰 길이를 계산하고 안정적인 통계를 반환합니다."""
    
    # 텍스트 전처리: 불필요한 줄바꿈이나 태그 제거
    cleaned_texts = text_series.apply(
        lambda x: str(x).replace('\n', ' ').replace('<br>', ' ')
    )
    # 토큰 길이 측정
    token_lengths = [len(tokenizer.tokenize(text)) for text in cleaned_texts]
    
    # ⚠️ Numpy를 사용해 95%와 99% 값을 안정적으로 계산
    lengths = np.array(token_lengths)

    # 길이가 0인 경우를 대비해 예외 처리
    if len(lengths) == 0:
        return {
            'count': 0, 'mean': 0, 'std': 0, 'min': 0, 
            '50%': 0, '95%': 0, '99%': 0, 'max': 0
        }

    # 통계 계산
    stats = {
        'count': len(lengths),
        'mean': round(np.mean(lengths), 3),
        'std': round(np.std(lengths), 3),
        'min': int(np.min(lengths)),
        '50%': int(np.percentile(lengths, 50)), # 중앙값
        '95%': int(np.percentile(lengths, 95)), # 95% 지점
        '99%': int(np.percentile(lengths, 99)), # 99% 지점 (가장 중요)
        'max': int(np.max(lengths))
    }
    
    return stats

def run_full_eda(data_path: str):
    """Train, Dev, Test 데이터셋의 대화 및 요약 토큰 길이를 분석합니다."""
    print("✨ 전체 데이터셋 (Train, Dev, Test) 토큰 길이 EDA 시작...")
    
    # 파일 경로가 'notebooks/data/'에 있을 경우를 대비하여 상대 경로를 지정합니다.
    # 일반적으로 대회 환경에서는 './data/'를 사용하지만, 혹시 문제가 생기면 'notebooks/data/'로 수정해야 할 수도 있어.
    
    datasets = {
        "train": os.path.join(data_path, 'train.csv'),
        "dev": os.path.join(data_path, 'dev.csv'),
        "test": os.path.join(data_path, 'test.csv')
    }
    
    all_stats = {}
    
    for name, file_path in datasets.items():
        if not os.path.exists(file_path):
            # 파일이 없으면, 다른 경로인 'notebooks/data/'를 시도해 볼 수도 있지만, 
            # 일단 사용자 경로인 './data/'를 사용한다고 가정할게.
            print(f"⚠️ 파일 경로를 찾을 수 없음: {file_path}. 데이터 경로를 확인해 주세요.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            print(f"\n--- {name.upper()} 데이터셋 ({len(df)}건) 분석 중... ---")

            # 1. Dialogue Length Analysis (인코더 길이 결정)
            dialogue_stats = calculate_token_lengths(df['dialogue'], tokenizer)
            all_stats[f'{name}_dialogue'] = dialogue_stats
            
            # 2. Summary Length Analysis (디코더 길이 결정)
            if 'summary' in df.columns:
                summary_stats = calculate_token_lengths(df['summary'], tokenizer)
                all_stats[f'{name}_summary'] = summary_stats
            else:
                print(f"[{name.upper()}] 요약(Summary) 컬럼은 테스트 데이터셋에 없으므로 생략합니다.")
        
        except Exception as e:
            print(f"❌ {name.upper()} 데이터 분석 중 오류 발생: {e}")
            
    print("\n=======================================================")
    print("✅ 전체 데이터셋 토큰 길이 EDA 결과:")
    print("=======================================================")
    
    # 결과를 보기 좋게 출력
    for key, stats in all_stats.items():
        print(f"\n[📊 {key.upper()} 통계]")
        pprint(stats)

# --- 메인 실행 ---
if __name__ == "__main__":
    run_full_eda(DATA_PATH)
