# TPU 지원을 위한 유틸리티 함수
import os
import torch
import logging

def is_tpu_available():
    """TPU 사용 가능 여부를 확인합니다."""
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False

def setup_tpu():
    """TPU 환경을 설정합니다."""
    if not is_tpu_available():
        logging.warning("TPU를 사용할 수 없습니다. PyTorch XLA가 설치되어 있는지 확인하세요.")
        return None
    
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    
    # TPU 슬라이싱 관련 환경 변수 설정
    os.environ['XLA_USE_BF16'] = '1'  # BF16 사용 (TPU에서 성능 향상)
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'  # 메모리 할당 크기 증가
    
    logging.info("TPU 환경을 설정합니다.")
    return {
        'xm': xm,
        'xmp': xmp,
        'pl': pl
    }

def get_xla_device():
    """XLA 디바이스를 반환합니다."""
    if not is_tpu_available():
        return None
    
    import torch_xla.core.xla_model as xm
    return xm.xla_device()

def get_device_type():
    """사용 가능한 디바이스 유형을 반환합니다 (tpu, cuda, cpu)."""
    if is_tpu_available():
        return "tpu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def create_xla_optimizer(optimizer):
    """XLA 최적화된 옵티마이저를 생성합니다."""
    if not is_tpu_available():
        return optimizer
    
    import torch_xla.core.xla_model as xm
    return xm.optimizer_step(optimizer)

def move_to_device(tensor, device):
    """텐서를 지정된 디바이스로 이동합니다."""
    if tensor is None:
        return None
    if isinstance(tensor, (list, tuple)):
        return [move_to_device(t, device) for t in tensor]
    if isinstance(tensor, dict):
        return {k: move_to_device(v, device) for k, v in tensor.items()}
    return tensor.to(device)

def create_parallel_loader(dataloader, device):
    """TPU에 최적화된 병렬 데이터 로더를 생성합니다."""
    if not is_tpu_available():
        return dataloader
    
    import torch_xla.distributed.parallel_loader as pl
    return pl.MpDeviceLoader(dataloader, device)


def get_tpu_cores_count():
    """사용 가능한 TPU 코어 수를 반환합니다."""
    if not is_tpu_available():
        return 0
    
    # TPU v2/v3는 일반적으로 8개 코어, TPU v4는 4개 코어
    # 환경 변수로 설정된 경우 해당 값을 사용
    if 'TPU_NUM_CORES' in os.environ:
        return int(os.environ['TPU_NUM_CORES'])
    
    # 기본값으로 8 반환 (TPU v2/v3 기준)
    return 8

def setup_tpu_slicing():
    """TPU 슬라이싱 환경을 설정합니다."""
    if not is_tpu_available():
        logging.warning("TPU를 사용할 수 없습니다. PyTorch XLA가 설치되어 있는지 확인하세요.")
        return None
    
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    
    # TPU 슬라이싱 관련 환경 변수 설정
    os.environ['XLA_USE_BF16'] = '1'  # BF16 사용 (TPU에서 성능 향상)
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'  # 메모리 할당 크기 증가
    
    logging.info("TPU 슬라이싱 환경을 설정합니다.")
    return {
        'xm': xm,
        'xmp': xmp,
        'pl': pl
    }

def sync_tpu_cores():
    """TPU 코어 간 동기화를 수행합니다."""
    if not is_tpu_available():
        return
    
    import torch_xla.core.xla_model as xm
    xm.mark_step()

def create_tpu_data_sampler(dataset, batch_size, rank=None, world_size=None, shuffle=True):
    """TPU에 최적화된 데이터 샘플러를 생성합니다."""
    if not is_tpu_available():
        return None
    
    import torch_xla.core.xla_model as xm
    from torch.utils.data.distributed import DistributedSampler
    
    if rank is None:
        rank = xm.get_ordinal()
    if world_size is None:
        world_size = xm.xrt_world_size()
    
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

if __name__ == "__main__":
    print(f"TPU 사용 가능 여부: {is_tpu_available()}")
    print(f"사용 가능한 디바이스 유형: {get_device_type()}")
    print(f"XLA 디바이스: {get_xla_device()}")
    if is_tpu_available():
        print(f"TPU 코어 수: {get_tpu_cores_count()}")
    