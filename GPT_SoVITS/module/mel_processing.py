import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


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
    key = "%s-%s-%s-%s-%s" %(dtype_device,n_fft, sampling_rate, hop_size, win_size)
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    # TPU 호환성을 위해 return_complex=False로 설정하고 명시적으로 실수 텐서로 처리
    try:
        # PyTorch 1.7+ 방식으로 시도
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[key],
                        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        
        # 명시적으로 실수/허수 부분 추출 (마지막 차원이 2인 실수 텐서)
        if spec.dim() > 0 and spec.shape[-1] == 2:
            spec_real, spec_imag = spec[..., 0], spec[..., 1]
            spec = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-9)
        else:
            # 이미 magnitude인 경우
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    except Exception as e:
        print(f"STFT error: {e}, trying alternative method")
        # 대체 방법: 명시적으로 실수/허수 부분 분리
        spec_complex = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[key],
                        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec_real = torch.real(spec_complex)
        spec_imag = torch.imag(spec_complex)
        spec = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-9)
    
    # 항상 float32로 변환하여 XLA 호환성 보장
    spec = spec.to(torch.float32)
        
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    # fmax_dtype_device = str(fmax) + '_' + dtype_device
    key = "%s-%s-%s-%s-%s-%s"%(dtype_device,n_fft, num_mels, sampling_rate, fmin, fmax)
    # if fmax_dtype_device not in mel_basis:
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        # mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
        mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    # spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
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
    # fmax_dtype_device = str(fmax) + '_' + dtype_device
    fmax_dtype_device = "%s-%s-%s-%s-%s-%s-%s-%s"%(dtype_device,n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
    # wnsize_dtype_device = str(win_size) + '_' + dtype_device
    wnsize_dtype_device = fmax_dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # TPU 호환성을 위해 return_complex=False로 설정하고 명시적으로 실수 텐서로 처리
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    print(f"spec dtype: {spec.dtype}, device: {spec.device}")
    # 명시적으로 실수/허수 부분 추출 (마지막 차원이 2인 실수 텐서)
    if spec.dim() > 0 and spec.shape[-1] == 2:
        spec_real, spec_imag = spec[..., 0], spec[..., 1]
        spec = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-9)
    else:
        # 이미 magnitude인 경우
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)

    
    # 항상 float32로 변환하여 XLA 호환성 보장
    spec = spec.to(torch.float32)
    # mel_basis도 float32로 변환하여 일관성 유지
    mel_basis_float32 = mel_basis[fmax_dtype_device].to(torch.float32)
    spec = torch.matmul(mel_basis_float32, spec)
    spec = spectral_normalize_torch(spec)
    # 최종 결과를 명시적으로 float32로 변환
    spec = spec.to(torch.float32) # XLA에게 타입 힌트를 주기 위해 float32로 변환
    assert not torch.is_complex(spec), f"spec is complex! dtype: {spec.dtype} : info {spec}"
    # 추가 검증: 텐서가 float32 타입인지 확인
    assert spec.dtype == torch.float32, f"spec is not float32! dtype: {spec.dtype}"
    return spec