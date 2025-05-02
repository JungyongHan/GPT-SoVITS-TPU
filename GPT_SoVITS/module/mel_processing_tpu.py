import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import logging

MAX_WAV_VALUE = 32768.0

# TPU 호환성을 위한 모듈 가져오기
sys_path_added = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    is_tpu = True
except ImportError:
    is_tpu = False

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.2:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.2:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    key = "%s-%s-%s-%s-%s" %(dtype_device, n_fft, sampling_rate, hop_size, win_size)
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    # TPU 호환성을 위한 STFT 처리 수정
    if is_tpu:
        # TPU에서는 return_complex=False 사용 (기본값)
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)
        # 마지막 차원이 2인 실수 텐서로 반환됨 (실수부, 허수부)
        spec_real, spec_imag = spec[..., 0], spec[..., 1]
        # 파워 스펙트럼 계산 (실수부^2 + 허수부^2)
        spec = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-8)
    else:
        # GPU/CPU에서는 기존 방식 사용
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    key = "%s-%s-%s-%s-%s-%s"%(dtype_device, n_fft, num_mels, sampling_rate, fmin, fmax)
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.2:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.2:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = "%s-%s-%s-%s-%s-%s-%s-%s"%(dtype_device, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
    wnsize_dtype_device = fmax_dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # TPU 호환성을 위한 STFT 처리 수정
    if is_tpu:
        # TPU에서는 return_complex=False 사용 (기본값)
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)
        # 마지막 차원이 2인 실수 텐서로 반환됨 (실수부, 허수부)
        spec_real, spec_imag = spec[..., 0], spec[..., 1]
        # 파워 스펙트럼 계산 (실수부^2 + 허수부^2)
        spec = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-8)
    else:
        # GPU/CPU에서는 기존 방식 사용
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec