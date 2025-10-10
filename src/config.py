import os
import yaml


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as file:
            loaded_config = yaml.safe_load(file)
    except FileNotFoundError:        
        print(f"오류: 설정파일 '{config_path}'을(를) 찾을 수 없습니다.")

    return loaded_config
