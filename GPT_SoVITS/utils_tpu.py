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
    

def sync_tpu_cores():
    """TPU 코어 간 동기화를 수행하고 메모리를 최적화합니다."""
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

# TPU v4-32 메모리 최적화 함수 추가
def optimize_tpu_memory():
    """TPU 메모리를 최적화합니다."""
    if not is_tpu_available():
        return False
    

    # TPU 메모리 정리
    import torch_xla.core.xla_model as xm
    xm.mark_step()
    
    try:
        # 불필요한 텐서 캐시 정리
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
        logging.debug(f"TPU 메모리 최적화 중 오류 발생: {e}")
    
    return True

# TPU 환경에서 복소수 텐서 처리를 위한 유틸리티 함수
def tpu_safe_stft(y, n_fft, hop_length, win_length, window, center=False, pad_mode='reflect', normalized=False, onesided=True):
    """
    TPU 환경에서 안전하게 STFT를 수행하는 함수입니다.
    view_as_complex_copy 연산을 사용하지 않고 실수 텐서로 처리합니다.
    """
    # TPU 환경에서는 return_complex=False로 설정하여 실수 텐서 반환
    spec = torch.stft(y, n_fft, hop_length=hop_length, win_length=win_length, window=window,
                     center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided)
    
    # 실수부와 허수부 분리
    spec_real, spec_imag = spec[..., 0], spec[..., 1]
    
    # 파워 스펙트럼 계산 (실수부^2 + 허수부^2)
    power_spec = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-8)
    
    return power_spec

# TPU 환경에서 복소수 텐서 처리를 위한 안전한 mel 스펙트로그램 생성 함수
def tpu_safe_mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, mel_basis, hann_window, center=False):
    """
    TPU 환경에서 안전하게 mel 스펙트로그램을 생성하는 함수입니다.
    view_as_complex_copy 연산을 사용하지 않고 실수 텐서로 처리합니다.
    """
    # 패딩 적용
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    # TPU 안전 STFT 사용
    spec = tpu_safe_stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                        center=center, pad_mode='reflect', normalized=False, onesided=True)
    
    # mel 변환 및 정규화
    spec = torch.matmul(mel_basis, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5) * 1.0)
    
    return spec

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import numpy as np

MAX_WAV_VALUE = 32768.0

# Pre-compute mel basis for common configurations to avoid recomputation
def precompute_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, dtype=torch.float32, device='xla'):
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(mel).to(dtype=dtype, device=device)

# Pre-compute hann windows for common configurations
def precompute_hann_window(win_size, dtype=torch.float32, device='xla'):
    return torch.hann_window(win_size).to(dtype=dtype, device=device)

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    # Use clamp_min instead of clamp for better TPU compatibility
    return torch.log(torch.clamp_min(x, clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)

# Static dictionaries for caching - avoid dynamic dictionary growth
mel_basis = {}
hann_window = {}

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # Avoid conditional print statements for TPU efficiency
    # Instead, use torch.clamp to ensure values are in range
    y = torch.clamp(y, min=-1.2, max=1.2)
    
    # Create a fixed key for caching
    device_type = y.device.type
    dtype_name = str(y.dtype).split('.')[-1]
    key = f"{dtype_name}_{device_type}_{n_fft}_{sampling_rate}_{hop_size}_{win_size}"
    
    # Initialize window if not in cache
    if key not in hann_window:
        hann_window[key] = precompute_hann_window(win_size, dtype=y.dtype, device=y.device)
    
    # Use static padding sizes for TPU
    pad_left = int((n_fft-hop_size)/2)
    pad_right = int((n_fft-hop_size)/2)
    
    # Pad the input
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode='reflect')
    y = y.squeeze(1)
    
    # Compute STFT
    spec = torch.stft(
        y, 
        n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window[key],
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        onesided=True, 
        return_complex=False
    )
    
    # Calculate magnitude
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    return spec

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # Create a fixed key for caching
    device_type = spec.device.type
    dtype_name = str(spec.dtype).split('.')[-1]
    key = f"{dtype_name}_{device_type}_{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}"
    
    # Initialize mel basis if not in cache
    if key not in mel_basis:
        mel_basis[key] = precompute_mel_basis(
            sampling_rate, n_fft, num_mels, fmin, fmax, 
            dtype=spec.dtype, device=spec.device
        )
    
    # Apply mel filterbank
    mel_spec = torch.matmul(mel_basis[key], spec)
    mel_spec = spectral_normalize_torch(mel_spec)
    return mel_spec

def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # Clamp values for stability
    y = torch.clamp(y, min=-1.2, max=1.2)
    
    # Create fixed keys for caching
    device_type = y.device.type
    dtype_name = str(y.dtype).split('.')[-1]
    mel_key = f"{dtype_name}_{device_type}_{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}"
    win_key = f"{dtype_name}_{device_type}_{win_size}"
    
    # Initialize mel basis and window if not in cache
    if mel_key not in mel_basis:
        mel_basis[mel_key] = precompute_mel_basis(
            sampling_rate, n_fft, num_mels, fmin, fmax, 
            dtype=y.dtype, device=y.device
        )
    
    if win_key not in hann_window:
        hann_window[win_key] = precompute_hann_window(win_size, dtype=y.dtype, device=y.device)
    
    # Use static padding sizes
    pad_left = int((n_fft-hop_size)/2)
    pad_right = int((n_fft-hop_size)/2)
    
    # Pad the input
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode='reflect')
    y = y.squeeze(1)
    
    # Compute STFT
    spec = torch.stft(
        y, 
        n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window[win_key],
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        onesided=True, 
        return_complex=False
    )
    
    # Calculate magnitude
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    
    # Apply mel filterbank
    mel_spec = torch.matmul(mel_basis[mel_key], spec)
    mel_spec = spectral_normalize_torch(mel_spec)
    
    return mel_spec



if __name__ == "__main__":
    print(f"TPU 사용 가능 여부: {is_tpu_available()}")
    print(f"사용 가능한 디바이스 유형: {get_device_type()}")
    print(f"XLA 디바이스: {get_xla_device()}")
    if is_tpu_available():
        print(f"TPU 코어 수: {get_tpu_cores_count()}")
        optimize_tpu_memory()
    