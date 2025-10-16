import pandas as pd
from typing import List
from pathlib import Path

def prepare_augmented_data(file_names: List[str], sample_size: int) -> pd.DataFrame:
    """
    제공된 증강 학습 데이터를 로드, 병합 및 지정된 크기로 샘플링하여 반환한다.
    """
    all_data: List[pd.DataFrame] = []
    
    # 1. 모든 파일 로드 및 병합
    for file_name in file_names:
        # 파일이 현재 폴더가 아닌 'data/' 폴더에 있을 가능성이 높으므로 경로를 지정
        file_path = Path(f"data/{file_name}")
        if file_path.exists():
            print(f"-> {file_name} 파일 로드 중...")
            df = pd.read_csv(file_path)
            all_data.append(df)
        else:
            # 파일이 'data/'에 없으면 현재 폴더에서 다시 시도해봄
            file_path = Path(file_name)
            if file_path.exists():
                print(f"-> {file_name} 파일 로드 중...")
                df = pd.read_csv(file_path)
                all_data.append(df)
            else:
                print(f"⚠️ 경고: {file_name} 파일을 찾을 수 없어. 경로를 확인해.")


    if not all_data:
        print("❌ 로드된 데이터가 없어. 작업을 중단한다.")
        return pd.DataFrame()
    
    # 모든 데이터를 하나로 합침
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 총 {len(combined_df)}개의 데이터 병합 완료.")

    # 2. 필요한 컬럼 선택: 'dialogue'와 'summary' (혹은 'topic')을 사용해야 해.
    # 네 파일에는 'summary'와 'topic'이 모두 있으니, 'topic'을 선택할게.
    final_columns = ['dialogue', 'topic']
    
    # 컬럼이 존재하지 않으면 오류 방지를 위해 처리
    for col in final_columns:
        if col not in combined_df.columns:
             print(f"❌ 데이터에 필수 컬럼 '{col}'이 없어. 작업을 중단한다.")
             return pd.DataFrame()

    combined_df = combined_df[final_columns]
    
    # 3. 샘플링 (훈련 시간 절약을 위해)
    if len(combined_df) > sample_size:
        print(f"✂️ 훈련 시간을 위해 {len(combined_df)}개에서 {sample_size}개로 샘플링한다.")
        # 랜덤으로 샘플링하여 데이터의 편향을 줄임 (random_state=42는 재현성을 위한 값)
        final_train_df = combined_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print("👍 샘플링 크기보다 데이터가 적어. 전체 데이터를 사용한다.")
        final_train_df = combined_df

    print(f"✨ 최종 훈련 데이터 크기: {len(final_train_df)}개")
    return final_train_df

if __name__ == '__main__':
    # 팀원이 준 파일 이름 리스트
    augmented_files: List[str] = [
        "train_augmented_rp.csv",
        "train_augmented_ss.csv",
        "train_augmented_synthetic.csv",
    ]
    
    # 최종적으로 훈련에 사용할 데이터 크기 (5000개로 설정)
    TARGET_SAMPLE_SIZE: int = 5000 
    
    final_df = prepare_augmented_data(
        file_names=augmented_files, 
        sample_size=TARGET_SAMPLE_SIZE
    )
    
    # 훈련 스크립트가 바로 사용할 수 있도록 임시 파일로 저장
    if not final_df.empty:
        OUTPUT_PATH = Path("data/final_combined_train_data.csv")
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ 최종 준비 완료! '{OUTPUT_PATH}' 파일을 훈련 스크립트에 사용해.")