# -*- coding: utf-8 -*-
import sys, os, traceback, numpy as np, torch
from scipy.io import wavfile
import librosa
from time import time as ttime
import shutil

# 환경 변수 세팅 및 임포트
inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
from feature_extractor import cnhubert

opt_dir = os.environ.get("opt_dir")
cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPT_SoVITS.utils_tpu import is_tpu_available, get_device_type, get_xla_device, move_to_device

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available() and not is_tpu_available()
now_dir = os.getcwd()
sys.path.append(now_dir)
from tools.my_utils import load_audio, clean_path

# 디바이스 감지
device_type = get_device_type()
if device_type == "tpu":
    import torch_xla.core.xla_model as xm
    device = get_xla_device()
    print("TPU 디바이스를 사용합니다.")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("CUDA 디바이스를 사용합니다.")
else:
    device = "cpu"
    print("CPU 디바이스를 사용합니다.")

model = cnhubert.get_model()
if is_half and device_type != "tpu":
    model = model.half().to(device)
else:
    model = model.to(device)

# 디렉토리 생성
hubert_dir = f"{opt_dir}/4-cnhubert"
wav32dir = f"{opt_dir}/5-wav32k"
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(hubert_dir, exist_ok=True)
os.makedirs(wav32dir, exist_ok=True)

maxx = 0.95
alpha = 0.5
nan_fails = []

def my_save(fea, path):
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = f"{ttime()}{i_part}.pth"
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, os.path.join(dir, name))

# ----------- 개선: 배치 처리 함수 -----------
def process_batch(wav_items):
    # wav_items: [(wav_name, wav_path), ...]
    batch_audio = []
    batch_names = []
    batch_max = []
    for wav_name, wav_path in wav_items:
        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            print(f"{wav_name}-filtered,{tmp_max}")
            continue
        tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * tmp_audio
        tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)
        batch_audio.append(tmp_audio)
        batch_names.append(wav_name)
        batch_max.append(tmp_max)
    if not batch_audio:
        return
    # 배치 텐서로 변환 및 디바이스 이동
    tensor_wavs = torch.stack([torch.from_numpy(a) for a in batch_audio]).to(device)
    if is_half and device_type != "tpu":
        tensor_wavs = tensor_wavs.half()
    with torch.no_grad():
        ssl = model.model(tensor_wavs.unsqueeze(1))["last_hidden_state"].transpose(1, 2).cpu()
    for i, wav_name in enumerate(batch_names):
        if np.isnan(ssl[i].detach().numpy()).sum() != 0:
            nan_fails.append((wav_name, wav_items[i][1]))
            print("nan filtered:%s" % wav_name)
            continue
        wavfile.write(f"{wav32dir}/{wav_name}", 32000, (batch_audio[i] * 32768).astype("int16"))
        my_save(ssl[i], f"{hubert_dir}/{wav_name}.pt")

# ----------- 메인 루프: 배치 단위로 처리 -----------
BATCH_SIZE = 8 if device_type == "tpu" else 1  # TPU는 반드시 배치로!
with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip().split("\n")

wav_items = []
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
        wav_items.append((wav_name, wav_path))
        if len(wav_items) >= BATCH_SIZE:
            process_batch(wav_items)
            wav_items = []
    except Exception:
        print(line, traceback.format_exc())
if wav_items:
    process_batch(wav_items)

# nan_fails 재처리 (옵션)
if nan_fails and is_half:
    is_half = False
    model = model.float()
    for wav in nan_fails:
        try:
            process_batch([wav])
        except Exception:
            print(wav[0], traceback.format_exc())
