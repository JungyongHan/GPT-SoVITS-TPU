import math
import torch
import torch.nn.functional as F
import numpy as np
from module import commons
from module.tpu.utils import mark_step, tpu_safe_randn

def slice_segments(x, ids_str, segment_size=4):
    """TPU-optimized version of slice_segments"""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths, segment_size=4):
    """TPU-optimized version of rand_slice_segments"""
    b, d, t = x.size()
    
    # Ensure x_lengths is at least segment_size
    x_lengths = torch.clamp_min(x_lengths, segment_size)
    
    # Generate random indices with static shape for TPU
    max_offset = x_lengths - segment_size
    # Use a more TPU-friendly way to generate random indices
    rand_offset = torch.floor(torch.rand([b], device=x.device) * max_offset.to(torch.float32) + 0.5).to(torch.long)
    
    # Ensure indices are valid
    rand_offset = torch.clamp_max(rand_offset, torch.max(max_offset))
    
    ret = slice_segments(x, rand_offset, segment_size)
    
    # Mark step for TPU execution
    mark_step()
    
    return ret, rand_offset

def sequence_mask(length, max_length=None):
    """TPU-optimized version of sequence_mask"""
    if max_length is None:
        max_length = torch.max(length)
    
    # Ensure max_length is a static value for TPU
    max_length = torch.max(max_length, torch.tensor(1, device=length.device))
    
    # Create indices tensor with static shape
    ids = torch.arange(0, max_length, device=length.device, dtype=length.dtype)
    
    # Broadcast for comparison
    mask = ids < length.unsqueeze(1)
    
    return mask

def generate_path(duration, mask):
    """TPU-optimized version of generate_path"""
    b, _, t_x, t_y = mask.shape
    
    # Use cumsum which is well-optimized on TPU
    cum_duration = torch.cumsum(duration, -1)
    
    # Create path using broadcasted comparison operations
    cum_duration_flat = cum_duration.view(b, t_x, 1)
    path = torch.arange(t_y, device=duration.device).view(1, 1, t_y) < cum_duration_flat
    
    # Handle the first step specially for correct path generation
    path = torch.nn.functional.pad(path, (0, 0, 1, 0))
    path = path[:, 1:] ^ path[:, :-1]
    path = path.to(torch.float32) * mask
    
    # Mark step for TPU execution
    mark_step()
    
    return path

def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """TPU-optimized version of fused_add_tanh_sigmoid_multiply"""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    
    # Split channels for activation functions
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    
    # Multiply activations
    acts = t_act * s_act
    
    # Mark step for TPU execution
    mark_step()
    
    return acts

def convert_pad_shape(pad_shape):
    """Convert pad shape to match torch.nn.functional.pad requirements"""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def subsequent_mask(length):
    """TPU-optimized version of subsequent_mask"""
    mask = torch.triu(torch.ones(length, length, device=length.device), diagonal=1).to(torch.bool)
    return mask

def shift_1d(x):
    """TPU-optimized version of shift_1d"""
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))
    x = x[:, :, :-1]
    return x

def element_wise_product(x, y):
    """Element-wise product of two tensors with broadcasting"""
    return x * y