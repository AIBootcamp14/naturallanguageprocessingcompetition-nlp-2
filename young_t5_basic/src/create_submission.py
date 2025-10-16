import pandas as pd

# 파일 경로 (이전 단계에서 생성된 파일이야)
INPUT_FILE = 'data/test_topic_solar_augmented.csv'
OUTPUT_FILE = 'final_submission.csv'

print("1. 보강된 파일 로드 중...")
try:
    # 1. 보강된 파일을 불러와.
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"오류: 보강된 파일 '{INPUT_FILE}'을 찾을 수 없어. 이전 단계가 성공했는지 확인해.")
    exit()

# 2. 'fname' 열과 SOLAR가 생성한 새로운 'topic' 열만 선택해.
#    (이 'topic' 열이 바로 네가 원하는 상세한 요약, 즉 'summary'야!)
submission_df = df[['fname', 'topic']]

# 3. 대회 제출 형식에 맞게 'topic' 열의 이름을 'summary'로 바꿔.
submission_df = submission_df.rename(columns={'topic': 'summary'})

# 4. 최종 제출 파일을 저장해.
print(f"2. 최종 제출 파일 '{OUTPUT_FILE}' 생성 완료!")
submission_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ 제출 준비 완료! 'final_submission.csv' 파일을 업로드해.")