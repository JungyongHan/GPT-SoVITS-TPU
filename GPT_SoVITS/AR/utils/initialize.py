#!/usr/bin/env python3
"""Initialize modules for espnet2 neural networks."""

import torch
from typeguard import check_argument_types
import sys
import os

# TPU 지원을 위한 유틸리티 함수 가져오기
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils_tpu import is_tpu_available, get_xla_device, move_to_device


def initialize(model: torch.nn.Module, init: str, device=None):
    """Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Custom initialization routines can be implemented into submodules
    as function `espnet_initialization_fn` within the custom module.

    Args:
        model: Target.
        init: Method of initialization.
        device: 모델을 초기화할 디바이스 (None이면 현재 디바이스 사용)
    """
    assert check_argument_types()
    print("init with", init)
    
    # TPU 디바이스 확인 및 설정
    if device is None:
        if is_tpu_available():
            device = get_xla_device()
            print(f"TPU 디바이스를 사용하여 모델을 초기화합니다: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA 디바이스를 사용하여 모델을 초기화합니다: {device}")
        else:
            device = torch.device("cpu")
            print("CPU를 사용하여 모델을 초기화합니다")

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init)
    # bias init
    for name, p in model.named_parameters():
        if ".bias" in name and p.dim() == 1:
            p.data.zero_()
            
    # 모델을 지정된 디바이스로 이동
    return move_to_device(model, device)
