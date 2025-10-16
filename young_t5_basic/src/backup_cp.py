import os
import shutil
from typing import List

# --------------------------------------------------------------------------------------
# 체크포인트 백업 함수 (PEP 8, PEP 484 준수)
# --------------------------------------------------------------------------------------
def backup_checkpoints(
    output_dir: str = './',
    checkpoints_to_backup: List[str] = None,
    backup_folder_name: str = 'checkpoints_backup'
) -> None:
    """
    지정된 체크포인트 폴더들을 백업 폴더로 복사한다.

    Args:
        output_dir: 체크포인트가 저장된 기본 디렉터리 경로 (예: './').
        checkpoints_to_backup: 백업할 체크포인트 폴더 이름 목록 (예: ['checkpoint-7008', 'checkpoint-10123']).
        backup_folder_name: 백업 파일을 저장할 상위 폴더 이름.
    """
    if checkpoints_to_backup is None:
        checkpoints_to_backup = []
        
    # 백업 파일을 저장할 최종 경로 (예: ./checkpoints_backup/)
    backup_base_path: str = os.path.join(output_dir, backup_folder_name)
    
    # 백업 디렉터리가 없으면 생성
    if not os.path.exists(backup_base_path):
        os.makedirs(backup_base_path)
        print(f"✨ 백업 폴더 '{backup_base_path}' 생성 완료.")
    
    # 백업할 체크포인트들에 대해 반복
    for checkpoint_name in checkpoints_to_backup:
        src_path: str = os.path.join(output_dir, checkpoint_name)
        dst_path: str = os.path.join(backup_base_path, checkpoint_name)
        
        # 1. 원본 체크포인트 폴더가 존재하는지 확인
        if not os.path.isdir(src_path):
            print(f"⚠️ 경고: 원본 체크포인트 폴더 '{src_path}'를 찾을 수 없어 건너뛴다.")
            continue
            
        # 2. 백업 대상 폴더가 이미 있으면 덮어쓰기 위해 기존 폴더를 삭제
        # shutil.copytree는 기본적으로 대상 폴더가 있으면 오류를 발생시킴
        if os.path.exists(dst_path):
            print(f"🔄 기존 백업 폴더 '{dst_path}'가 존재하여 먼저 삭제한다.")
            shutil.rmtree(dst_path)
        
        try:
            # 3. 폴더 전체 복사 (재귀적으로 파일과 하위 폴더 모두 복사)
            # copytree는 디렉토리 트리를 통째로 복사하는 가장 좋은 방법이야.
            shutil.copytree(src_path, dst_path)
            print(f"✅ 체크포인트 '{checkpoint_name}' 백업 완료: -> '{dst_path}'")
        except Exception as e:
            print(f"❌ '{checkpoint_name}' 백업 중 오류 발생: {e}")


# --------------------------------------------------------------------------------------
# 백업 실행 예시 (메인 코드에 추가하지 않고 따로 실행할 때 사용)
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 네가 백업하고 싶은 체크포인트 목록
    checkpoints = ['checkpoint-10123']
    
    # 현재 디렉토리에서 'checkpoints_backup' 폴더에 백업 실행
    backup_checkpoints(
        output_dir='./',
        checkpoints_to_backup=checkpoints
    )
    print("\n💡 모든 요청 체크포인트 백업 시도 완료.")