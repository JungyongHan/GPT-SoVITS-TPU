import sys
import os
import torch
import numpy as np
from scipy.io import wavfile
import librosa
import traceback
from time import time as ttime
import shutil

# TPU 지원
from GPT_SoVITS.utils_tpu import is_tpu_available, get_device_type, get_xla_device, move_to_device

# 환경 변수
inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
opt_dir = os.environ.get("opt_dir")
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available() and not is_tpu_available()

from feature_extractor import cnhubert
cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")

now_dir = os.getcwd()
sys.path.append(now_dir)
from tools.my_utils import load_audio, clean_path

# 디바이스 감지
device_type = get_device_type()
if device_type == "tpu":
    device = get_xla_device()
    print("TPU 디바이스를 사용합니다.")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("CUDA 디바이스를 사용합니다.")
else:
    device = "cpu"
    print("CPU 디바이스를 사용합니다.")

# 디렉토리 생성
hubert_dir = f"{opt_dir}/4-cnhubert"
wav32dir = f"{opt_dir}/5-wav32k"
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(hubert_dir, exist_ok=True)
os.makedirs(wav32dir, exist_ok=True)

maxx = 0.95
alpha = 0.5

model = cnhubert.get_model()
# 모델은 한 번만 device로 이동
if is_half and device_type != "tpu":
    model = model.half().to(device)
else:
    model = model.to(device)

model.eval()
nan_fails = []

# TPU/XLA: 연산 후 mark_step() 필요
if device_type == "tpu":
    import torch_xla.core.xla_model as xm

def my_save(fea, path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = f"{ttime()}{i_part}.pth"
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, f"{dir}/{name}")

def name2go(wav_name, wav_path):
    hubert_path = f"{hubert_dir}/{wav_name}.pt"
    if os.path.exists(hubert_path):
        return
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print(f"{wav_name}-filtered,{tmp_max}")
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * tmp_audio
    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if is_half and device_type != "tpu":
        tensor_wav16 = tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    with torch.no_grad():
        ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
        if device_type == "tpu":
            xm.mark_step()  # TPU에서 연산 즉시 실행
    if np.isnan(ssl.detach().numpy()).sum() != 0:
        nan_fails.append((wav_name, wav_path))
        print(f"nan filtered:{wav_name}")
        return
    wavfile.write(f"{wav32dir}/{wav_name}", 32000, tmp_audio32.astype("int16"))
    my_save(ssl, hubert_path)

# 파일 리스트 읽기
with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip("\n").split("\n")

for line in lines[int(i_part)::int(all_parts)]:
    try:
        wav_name, spk_name, language, text = line.split("|")
        wav_name = clean_path(wav_name)
        if inp_wav_dir:
            wav_name = os.path.basename(wav_name)
            wav_path = f"{inp_wav_dir}/{wav_name}"
        else:
            wav_path = wav_name
            wav_name = os.path.basename(wav_name)
        name2go(wav_name, wav_path)
    except:
        print(line, traceback.format_exc())

if len(nan_fails) > 0 and is_half:
    is_half = False
    model = model.float()
    for wav in nan_fails:
        try:
            name2go(wav[0], wav[1])
        except:
            print(wav[0], traceback.format_exc())
