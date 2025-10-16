import pandas as pd
import os


file_paths = [
    './data/train_augmented_rp.csv',
    './data/train_augmented_ss.csv',
    './data/train_augmented_synthetic.csv'
]

# 모든 파일을 읽어 리스트에 저장
all_dfs = []
for path in file_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # 'Unnamed:' 컬럼이 있다면 제거 
        cols_to_drop = [col for col in df.columns if col.startswith('Unnamed')]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"✅ {path}: 불필요한 컬럼 {cols_to_drop} 제거 완료.")

        # 필수 컬럼 4개(fname, dialogue, summary, topic)가 모두 있는지 확인
        required_cols = {'fname', 'dialogue', 'summary', 'topic'}
        if required_cols.issubset(df.columns):
            all_dfs.append(df[list(required_cols)])
        else:
            # 이 경고가 뜨면 파일 중 하나가 이미 깨진 거
            print(f"❌ 경고: {path} 파일에 필수 컬럼(summary)이 없어 제외합니다. 현재 컬럼: {df.columns.tolist()}")
    else:
        print(f"❌ 경고: {path} 파일이 존재하지 않습니다. 경로를 확인하세요.")

# 모든 데이터프레임을 하나로 합침
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 최종 파일 저장 (인덱스 저장 금지: index=False가 핵심!)
    output_path = './data/final_combined_train_data.csv'
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n🎉 성공: 총 {len(combined_df)}개의 데이터로 {output_path} 파일 재작성 완료!")
else:
    print("\n❌ 실패: 합칠 수 있는 유효한 파일이 없어 훈련을 진행할 수 없습니다. 원본 증강 파일을 확인하세요.")