"""Prepare training data from train.csv and dev.csv"""
import pandas as pd
from datasets import Dataset


def format_prompt(dialogue: str, summary: str = None) -> dict:
    """Solar-Pro 형식에 맞는 프롬프트 생성"""
    messages = [
        {
            "role": "user",
            "content": f"다음 대화를 요약하세요.\n\n{dialogue}\n\n요약:"
        }
    ]

    if summary:
        messages.append({
            "role": "assistant",
            "content": summary
        })

    return {"messages": messages}


def prepare_dataset(csv_path: str) -> Dataset:
    """CSV를 HuggingFace Dataset으로 변환"""
    df = pd.read_csv(csv_path)

    # fname, dialogue, summary 컬럼만 사용
    df = df[['fname', 'dialogue', 'summary']].copy()

    # 빈 요약 제거
    df = df[df['summary'].str.strip() != ''].copy()

    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Avg dialogue length: {df['dialogue'].str.len().mean():.0f} chars")
    print(f"Avg summary length: {df['summary'].str.len().mean():.0f} chars")

    # 프롬프트 형식으로 변환
    formatted = []
    for _, row in df.iterrows():
        formatted.append(format_prompt(row['dialogue'], row['summary']))

    return Dataset.from_list(formatted)


if __name__ == "__main__":
    # Train 및 Dev 데이터 로드
    train_dataset = prepare_dataset("data/train.csv")
    dev_dataset = prepare_dataset("data/dev.csv")

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Dev: {len(dev_dataset)} samples")

    # 샘플 확인
    print("\n=== Sample ===")
    print(train_dataset[0])

    # 저장
    train_dataset.save_to_disk("data/train_formatted")
    dev_dataset.save_to_disk("data/dev_formatted")
    print("\n✓ Saved to data/train_formatted and data/dev_formatted")
