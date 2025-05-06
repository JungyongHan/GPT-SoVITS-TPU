import torch
import torch.nn.functional as F
import math

# Check if XLA is available and import it
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

def get_xla_device():
    """Get the XLA device if available, otherwise return CPU device"""
    if HAS_XLA:
        return xm.xla_device()
    return torch.device('cpu')

def tpu_random_seed(seed=None):
    """Set random seed for TPU operations"""
    if HAS_XLA:
        xm.set_rng_state(seed)
    torch.manual_seed(seed if seed is not None else 42)

def tpu_safe_randn(*args, **kwargs):
    """Generate random numbers in a TPU-compatible way"""
    # TPUs work better with statically shaped tensors
    # This function ensures random tensor generation is TPU-friendly
    device = kwargs.pop('device', None)
    dtype = kwargs.pop('dtype', None)
    
    # Generate on CPU first for better compatibility
    tensor = torch.randn(*args, **kwargs)
    
    # Then move to the appropriate device
    if device is not None:
        tensor = tensor.to(device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    
    return tensor

def tpu_safe_cat(tensors, dim=0):
    """Concatenate tensors in a TPU-friendly way"""
    # Ensure all tensors have the same device
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        # Move all tensors to the same device (the first tensor's device)
        target_device = tensors[0].device
        tensors = [t.to(target_device) for t in tensors]
    
    return torch.cat(tensors, dim=dim)

def tpu_safe_split(tensor, split_size_or_sections, dim=0):
    """Split a tensor in a TPU-friendly way"""
    # TPUs prefer static shapes, so we ensure the split operation
    # results in well-defined shapes
    return torch.split(tensor, split_size_or_sections, dim)

def mark_step():
    """Mark a step for TPU execution"""
    if HAS_XLA:
        xm.mark_step()