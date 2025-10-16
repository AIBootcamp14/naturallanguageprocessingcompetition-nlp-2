# src/data/preprocess/typo_fix.py
import pandas as pd
import os
import re
from tqdm import tqdm
from hanspell import spell_checker
import xml.etree.ElementTree as ET

# tqdm을 pandas에 적용하기 위한 설정
tqdm.pandas()

# --- 설정 (상수) ---
INPUT_DIR = 'data/clean/denoise/'
OUTPUT_DIR = 'data/clean/typo_fix/'
FILES_TO_PROCESS = ['train.csv', 'dev.csv', 'test.csv']
MASK_PATTERN = re.compile(r'(#[A-Za-z0-9_]+#?:?)')


def correct_dialogue_safely(dialogue):
    """
    마스킹 토큰을 보호하면서 대화 스크립트의 맞춤법을 교정하는 함수
    """
    if not isinstance(dialogue, str):
        return ""  # NaN이나 다른 타입의 데이터일 경우 빈 문자열 반환

    try:
        # 1. 마스킹 토큰을 찾아서 리스트에 저장하고, 텍스트에서는 placeholder로 교체
        original_masks = MASK_PATTERN.findall(dialogue)
        placeholder = "__MASK__"
        masked_dialogue = MASK_PATTERN.sub(placeholder, dialogue)

        # 2. 맞춤법 교정 로직 (줄바꿈 단위로 처리)
        lines = masked_dialogue.split('\n')
        corrected_lines = []
        for line in lines:
            if line.strip():
                try:
                    result = spell_checker.check(line)
                    corrected_lines.append(result.checked)
                except (ET.ParseError, Exception):
                    corrected_lines.append(line)  # 교정 실패 시 원본 줄 사용
            else:
                corrected_lines.append(line)

        corrected_dialogue = '\n'.join(corrected_lines)

        # 3. 교정된 텍스트의 placeholder를 원래 마스킹 토큰으로 순서대로 복원
        for mask in original_masks:
            corrected_dialogue = corrected_dialogue.replace(
                placeholder, mask, 1)

        return corrected_dialogue

    except Exception as e:
        # 전체 프로세스에서 에러 발생 시 원본 대화 반환
        print(f"처리 중 예외 발생: {e}")
        return dialogue


def main():
    """
    메인 실행 함수: 파일들을 순회하며 맞춤법 교정 전처리를 수행합니다.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과 저장 폴더: '{OUTPUT_DIR}'")

    for filename in FILES_TO_PROCESS:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(input_path):
            print(f"파일을 찾을 수 없습니다: {input_path}")
            continue

        print(f"\n--- {filename} 처리 시작 ---")

        # CSV 파일 불러오기
        df = pd.read_csv(input_path)

        # dialogue 컬럼에 맞춤법 교정 함수 적용 (tqdm으로 진행 상황 표시)
        df['dialogue'] = df['dialogue'].progress_apply(correct_dialogue_safely)

        # 수정된 데이터프레임을 새로운 CSV 파일로 저장
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"[처리 완료] {filename} → 결과 저장: {output_path}")

    print("\n모든 파일 처리가 완료되었습니다.")


if __name__ == "__main__":
    main()
