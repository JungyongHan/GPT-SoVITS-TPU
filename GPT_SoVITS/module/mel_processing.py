import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

# TPU 환경 감지 함수
def is_tpu_available():
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
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
    
    # TPU 환경에서는 매번 새로 생성하여 디바이스 일치 보장
    if is_tpu_available() or key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    
    # TPU에서는 window가 올바른 디바이스에 있는지 확인
    if hann_window[key].device != y.device:
        hann_window[key] = hann_window[key].to(device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    # TPU 환경에서는 return_complex=True 사용 권장
    if is_tpu_available():
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.abs(spec)
    else:
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    
    return spec

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    key = "%s-%s-%s-%s-%s-%s"%(dtype_device, n_fft, num_mels, sampling_rate, fmin, fmax)
    
    # TPU 환경에서는 매번 새로 생성하거나 디바이스 확인
    if is_tpu_available() or key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    
    # TPU에서는 mel_basis가 올바른 디바이스에 있는지 확인
    if mel_basis[key].device != spec.device:
        mel_basis[key] = mel_basis[key].to(device=spec.device)
        
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
    
    # TPU 환경에서는 매번 새로 생성하거나 디바이스 확인
    if is_tpu_available() or fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    
    if is_tpu_available() or fmax_dtype_device not in hann_window:
        hann_window[fmax_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    
    # TPU에서는 텐서들이 올바른 디바이스에 있는지 확인
    if mel_basis[fmax_dtype_device].device != y.device:
        mel_basis[fmax_dtype_device] = mel_basis[fmax_dtype_device].to(device=y.device)
    
    if hann_window[fmax_dtype_device].device != y.device:
        hann_window[fmax_dtype_device] = hann_window[fmax_dtype_device].to(device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    # TPU 환경에서는 return_complex=True 사용 권장
    if is_tpu_available():
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[fmax_dtype_device],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.abs(spec)
    else:
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[fmax_dtype_device],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
