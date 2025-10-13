import torch
import gc

def clean_gpu_memory():
    # GPU 메모리 완전 초기화
    torch.cuda.empty_cache()
    gc.collect()

    # 모든 CUDA 텐서 삭제
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except:
            pass

    gc.collect()
    torch.cuda.empty_cache()
    
    torch.backends.cuda.matmul.allow_tf32 = True

    # 1. 현재 할당된 메모리 (PyTorch가 사용 중)
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"Allocated: {allocated:.2f} GB")  # 사용 중인 메모리

    # 2. 예약된 메모리 (PyTorch가 확보해둔 메모리)
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Reserved: {reserved:.2f} GB")    # 캐시 포함

    # 3. 전체 GPU 메모리
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total: {total:.2f} GB")          # GPU 전체 용량

    # 4. 사용 가능한 메모리 (대략)
    free = total - reserved
    print(f"Free: {free:.2f} GB")
    
    # 5. 서버 자체의 실제 남은 메모리
    free, total = torch.cuda.mem_get_info()  # 바이트 단위, "디바이스" 잔여
    print(f"server'free={free/1024**3:.2f} GB / total={total/1024**3:.2f} GB")