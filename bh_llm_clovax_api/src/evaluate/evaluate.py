"""평가 지표 계산 자리표시자."""
from pathlib import Path
from typing import Sequence

def evaluate(predictions: Sequence[int], references: Sequence[int]) -> float:
    """간단한 정확도 지표를 계산합니다."""
    if not predictions:
        return 0.0
    hits = sum(int(p == r) for p, r in zip(predictions, references))
    return hits / len(predictions)


def main() -> None:
    """CLI에서 사용될 기본 함수."""
    print("[EVAL] 평가 스텁을 실행했습니다.")


if __name__ == "__main__":
    main()
