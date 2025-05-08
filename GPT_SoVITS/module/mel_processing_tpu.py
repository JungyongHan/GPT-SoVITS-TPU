import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import torch_xla.core.xla_model as xm
import torch_xla
import functools


MAX_WAV_VALUE = 32768.0

# 캐시 딕셔너리 - 메모리 사용량 최적화
mel_basis = {}
hann_window = {}

# 동적 범위 압축 함수
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 스펙트럼 정규화 함수
def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

# 스펙트로그램 계산 함수 - 컴파일 가능한 버전
def spectrogram_torch_tpu(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # 캐시 키 생성
    dtype_device = str(y.dtype) + '_' + str(y.device)
    key = f"{dtype_device}-{n_fft}-{sampling_rate}-{hop_size}-{win_size}"
    
    # 한 윈도우 캐싱
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    
    # 패딩 적용
    y = torch.nn.functional.pad(
        y.unsqueeze(1), 
        (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), 
        mode='reflect'
    )
    y = y.squeeze(1)
    
    # STFT 계산
    spec = torch.stft(
        y, n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window[key],
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        onesided=True, 
        return_complex=False
    )
    
    # 스펙트럼 크기 계산
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    return spec

# 스펙트럼을 멜 스펙트럼으로 변환하는 함수 - 컴파일 가능한 버전
def spec_to_mel_torch_tpu(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # 캐시 키 생성
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    key = f"{dtype_device}-{n_fft}-{num_mels}-{sampling_rate}-{fmin}-{fmax}"
    
    # 멜 필터뱅크 캐싱
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    
    # 역전파 최적화를 위한 메모리 사용량 감소
    with torch.no_grad():
        # 중간 계산 결과를 미리 계산하여 메모리 사용량 감소
        mel_filter = mel_basis[key]
    
    # 멜 스펙트럼 계산 - 역전파 최적화
    # 행렬 곱셈 연산 최적화 (TPU에서 더 효율적인 연산 사용)
    mel_spec = torch.matmul(mel_filter, spec)
    
    # 스펙트럼 정규화
    mel_spec = spectral_normalize_torch(mel_spec)
        
    return mel_spec

# 멜 스펙트로그램 계산 함수 - 컴파일 가능한 버전
def mel_spectrogram_torch_tpu(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # TPU에서 역전파 최적화를 위한 메모리 사용량 감소
    # 스펙트로그램 계산
    spec = spectrogram_torch_tpu(y, n_fft, sampling_rate, hop_size, win_size, center)
    
    # 중간 계산 결과 체크포인트 설정 (메모리 최적화)
    if hasattr(torch, 'utils') and hasattr(torch.utils, 'checkpoint') and torch.is_grad_enabled():
        # 그래디언트 체크포인트 사용 (메모리 사용량 감소, 역전파 시간 약간 증가)
        mel_spec = torch.utils.checkpoint.checkpoint(
            spec_to_mel_torch_tpu, spec, n_fft, num_mels, sampling_rate, fmin, fmax
        )
    else:
        # 일반 계산
        mel_spec = spec_to_mel_torch_tpu(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
            
    return mel_spec

# 컴파일된 함수 생성
try:
    # 최신 PyTorch-XLA API를 사용하여 함수 컴파일
    # 컴파일 옵션 설정 - 메모리 사용량 최적화
    compile_options = {
        "backend": "xla",
        "fullgraph": True,
        "min_compilation_time": 0,  # 즉시 컴파일
        "memory_optimization_level": 2,  # 메모리 최적화 수준 (0-2)
        "allow_xla_auto_fusion": True  # XLA 자동 퓨전 허용
    }
    
    # 함수 컴파일
    compiled_spec_to_mel = torch_xla.compile(
        spec_to_mel_torch_tpu,
        **compile_options,
        name="spec_to_mel"
    )
    
    compiled_mel_spectrogram = torch_xla.compile(
        mel_spectrogram_torch_tpu,
        **compile_options,
        name="mel_spectrogram"
    )
    
    # 컴파일된 함수 사용 - 캐싱 메커니즘 추가
    _mel_cache = {}
    _spec_cache = {}
    
    def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
        # 캐시 키 생성
        cache_key = f"{spec.shape}_{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}"
        
        # 캐시에 있으면 재사용
        if cache_key in _mel_cache and _mel_cache[cache_key].shape == spec.shape[:-1] + (num_mels,):
            return _mel_cache[cache_key]
            
        # 없으면 계산하고 캐시에 저장
        result = compiled_spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
            
        # 캐시 크기 제한 (최대 5개 항목만 유지)
        if len(_mel_cache) > 5:
            # 가장 오래된 항목 제거
            _mel_cache.pop(next(iter(_mel_cache)))
            
        _mel_cache[cache_key] = result
        return result
    
    def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        # 캐시 키 생성
        cache_key = f"{y.shape}_{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{center}"
        
        # 캐시에 있으면 재사용
        if cache_key in _spec_cache and _spec_cache[cache_key].shape == y.shape[:-1] + (num_mels,):
            return _spec_cache[cache_key]
            
        # 없으면 계산하고 캐시에 저장
        result = compiled_mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center)
            
        # 캐시 크기 제한 (최대 5개 항목만 유지)
        if len(_spec_cache) > 5:
            # 가장 오래된 항목 제거
            _spec_cache.pop(next(iter(_spec_cache)))
            
        _spec_cache[cache_key] = result
        return result

except Exception as e:
    # 컴파일에 실패한 경우 원래 함수 사용
    xm.master_print(f"멜 처리 함수 컴파일 실패: {e}")
    xm.master_print("기본 함수를 사용합니다.")
    
    # 기본 함수 정의
    def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
        return spec_to_mel_torch_tpu(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    
    def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        return mel_spectrogram_torch_tpu(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center)