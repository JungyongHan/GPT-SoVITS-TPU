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
    return pl.ParallelLoader(dataloader, [device]).per_device_loader(device)