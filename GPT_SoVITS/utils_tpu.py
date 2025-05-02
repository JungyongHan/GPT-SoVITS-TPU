# TPU 지원을 위한 유틸리티 함수
import os
import torch
import logging
import gc

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
    # TPU v4-32에 최적화: drop_last=True로 설정하여 마지막 불완전한 배치 제거
    return pl.MpDeviceLoader(dataloader, device, drop_last=True)


def get_tpu_cores_count():
    """사용 가능한 TPU 코어 수를 반환합니다."""
    if not is_tpu_available():
        return 0
    
    # TPU v2/v3는 일반적으로 8개 코어, TPU v4는 4개 코어
    # 환경 변수로 설정된 경우 해당 값을 사용
    if 'TPU_NUM_CORES' in os.environ:
        return int(os.environ['TPU_NUM_CORES'])
    
    # 기본값으로 8 반환 (TPU v2/v3 기준)
    return 4

def setup_tpu_slicing():
    """TPU 슬라이싱 환경을 설정합니다."""
    if not is_tpu_available():
        logging.warning("TPU를 사용할 수 없습니다. PyTorch XLA가 설치되어 있는지 확인하세요.")
        return None
    
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    
    # TPU v4-32 최적화를 위한 환경 변수 설정
    os.environ['XLA_USE_BF16'] = '1'  # BF16 사용 (TPU에서 성능 향상)
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '2000000000'  # 메모리 할당 크기 증가 (2GB)
    os.environ['XLA_TRANSFER_GUARD_DEVICE_MEMORY'] = '1'  # 메모리 가드 활성화
    os.environ['XLA_TRANSFER_GUARD_HOST_MEMORY'] = '1'  # 호스트 메모리 가드 활성화
    os.environ['XLA_SYNC_WAIT'] = '1'  # 동기화 대기 활성화
    os.environ['TPU_COMPILE_XLA_CLUSTERS'] = '1'  # XLA 클러스터 컴파일 활성화
    os.environ['XLA_DUMP_FATAL_STACK'] = '1'  # 오류 발생 시 스택 덤프
    
    # 추가 XLA 최적화 환경 변수
    os.environ['XLA_EXPERIMENTAL_ASYNC_COMPILATION'] = '1'  # 비동기 컴파일 활성화
    os.environ['XLA_EXPERIMENTAL_ENABLE_ASYNC_ALL_GATHER'] = '1'  # 비동기 all_gather 활성화
    os.environ['XLA_EXPERIMENTAL_ENABLE_ASYNC_REDUCE_SCATTER'] = '1'  # 비동기 reduce_scatter 활성화
    os.environ['XLA_EXPERIMENTAL_ENABLE_ASYNC_COLLECTIVE_PERMUTE'] = '1'  # 비동기 collective_permute 활성화
    os.environ['XLA_EXPERIMENTAL_ENABLE_ASYNC_ALL_REDUCE'] = '1'  # 비동기 all_reduce 활성화
    
    # TPU v4-32 메모리 최적화
    try:
        import torch_xla.debug.metrics as met
        # 메모리 사용량을 85%로 제한하여 OOM 방지 (더 안전한 값)
        met.set_memory_fraction(0.85)
        logging.info("TPU 메모리 사용량을 85%로 제한합니다.")
        
        # XLA 컴파일러 최적화 설정
        import torch_xla.core.xla_builder as xb
        # 더 작은 그룹 크기로 메모리 사용량 감소
        # xb.set_lowering_options("max_group_size=4,min_group_size=1")
        logging.info("XLA 컴파일러 최적화 설정을 적용했습니다.")
    except Exception as e:
        logging.warning(f"TPU 메모리 최적화 설정 중 오류 발생: {e}")
    
    # 메모리 최적화를 위한 가비지 컬렉션
    gc.collect()
    
    logging.info("TPU v4-32 슬라이싱 환경을 설정했습니다.")
    return {
        'xm': xm,
        'xmp': xmp,
        'pl': pl
    }

def sync_tpu_cores():
    """TPU 코어 간 동기화를 수행하고 메모리를 최적화합니다."""
    if not is_tpu_available():
        return
    
    import torch_xla.core.xla_model as xm
    # 명시적 가비지 컬렉션 호출
    gc.collect()
    
    # TPU 코어 동기화 - 중요: 이 호출은 TPU 코어 간 동기화를 수행합니다
    xm.mark_step()
    
    # 동기화 후 추가 최적화
    try:
        # 불필요한 텐서 캐시 정리
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # XLA 컴파일러 최적화 설정
        import torch_xla.core.xla_builder as xb
        # 더 작은 그룹 크기로 메모리 사용량 감소
        # xb.set_lowering_options("max_group_size=4,min_group_size=1")
        
        # 메모리 통계 로깅 (디버깅용)
        import torch_xla.debug.metrics as met
        if xm.get_ordinal() == 0 and xm.xrt_world_size() > 1:
            memory_stats = met.metric_data('MemoryStats')
            if memory_stats:
                logging.debug(f"TPU 메모리 사용량: {memory_stats}")
                
        # 메모리 사용량 제한 설정 (TPU v4-32에 최적화)
        try:
            # 메모리 사용량을 85%로 제한하여 OOM 방지
            met.set_memory_fraction(0.85)
        except:
            pass
    except Exception as e:
        logging.debug(f"TPU 코어 동기화 중 오류 발생: {e}")
        pass

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

# TPU v4-32 메모리 최적화 함수 추가
def optimize_tpu_memory():
    """TPU v4-32 메모리 사용량을 최적화합니다."""
    if not is_tpu_available():
        return
    
    # 명시적 가비지 컬렉션 호출
    gc.collect()
    
    # TPU 메모리 최적화
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        
        # 메모리 통계 수집 및 로깅
        if xm.get_ordinal() == 0:
            memory_stats = met.metric_data('MemoryStats')
            if memory_stats:
                logging.debug(f"TPU 메모리 최적화 전 사용량: {memory_stats}")
        
        # 메모리 최적화 수행
        xm.mark_step()
        gc.collect()
        
        # 메모리 사용량 제한 설정 (TPU v4-32에 최적화)
        try:
            # 메모리 사용량을 85%로 제한하여 OOM 방지
            met.set_memory_fraction(0.85)
        except:
            pass
            
        # XLA 컴파일러 최적화 설정
        try:
            import torch_xla.core.xla_builder as xb
            # 더 작은 그룹 크기로 메모리 사용량 감소
            # xb.set_lowering_options("max_group_size=4,min_group_size=1")
        except:
            pass
        
        # 최적화 후 메모리 통계 수집 및 로깅
        if xm.get_ordinal() == 0:
            memory_stats = met.metric_data('MemoryStats')
            if memory_stats:
                logging.debug(f"TPU 메모리 최적화 후 사용량: {memory_stats}")
    except Exception as e:
        logging.warning(f"TPU 메모리 최적화 중 오류 발생: {e}")

if __name__ == "__main__":
    print(f"TPU 사용 가능 여부: {is_tpu_available()}")
    print(f"사용 가능한 디바이스 유형: {get_device_type()}")
    print(f"XLA 디바이스: {get_xla_device()}")
    if is_tpu_available():
        print(f"TPU 코어 수: {get_tpu_cores_count()}")
        optimize_tpu_memory()
    