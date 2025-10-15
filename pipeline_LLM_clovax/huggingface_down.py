
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    # 예: C:\tmp\clovax  또는 /data/models/clovax
    local_dir=r"C:\Users...",
    local_dir_use_symlinks=False,
    revision="main",
    # CLI로 로그인 못할 때, 코드에만 일시적으로 넣어 실행 후 제거
    token="1234123412314"
)
