"""학습 루틴 자리표시자."""
from pathlib import Path

def run_training(config_path: Path | str) -> None:
    """주어진 설정을 기반으로 학습을 수행합니다."""
    cfg_path = Path(config_path)
    print(f"[TRAIN] 설정 파일을 불러옵니다: {cfg_path}")
    # TODO: 데이터 로딩, 모델 정의, 학습 루프 구현


if __name__ == "__main__":
    run_training("configs/train.yaml")
