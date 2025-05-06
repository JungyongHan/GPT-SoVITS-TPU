# TPU Environment Setup for GPT-SoVITS

This guide will help you set up the necessary environment to run the TPU-optimized version of GPT-SoVITS.

## Prerequisites

- Python 3.8 or higher
- Access to a TPU device (Google Cloud TPU VM or Colab TPU runtime)

## Installation Steps

### 1. Install PyTorch and PyTorch/XLA

The TPU-optimized models require PyTorch/XLA 2.5.0 or higher. Install the compatible versions:

```bash
pip install torch==2.5.0
pip install torch_xla==2.5.0
```

For Google Colab TPU runtime, you can use:

```python
!pip install torch==2.5.0
!pip install torch_xla==2.5.0
```

### 2. Verify TPU Installation

Run the following Python code to verify that PyTorch/XLA is correctly installed and can detect your TPU device:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Get the XLA device
device = xm.xla_device()
print(f"XLA device: {device}")

# Create a test tensor
x = torch.randn(2, 3)
x = x.to(device)
print(f"Tensor on TPU: {x}")

# Perform a simple operation to verify TPU execution
y = x + x
print(f"Result: {y}")

# Mark step to execute operations
xm.mark_step()
```

### 3. Environment Variables

Set the following environment variables to optimize TPU performance:

```bash
export XLA_USE_BF16=1  # Enable bfloat16 for better performance
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000  # Increase tensor allocator size
export TF_CPP_MIN_LOG_LEVEL=0  # Control logging level
```

In Python, you can set these with:

```python
import os
os.environ['XLA_USE_BF16'] = '1'
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
```

### 4. Memory Management

TPUs have different memory characteristics than GPUs. To avoid out-of-memory errors:

- Use smaller batch sizes initially and gradually increase
- Enable gradient checkpointing for large models
- Use the `mark_step()` function at strategic points in your code
- Consider using mixed precision training with bfloat16

## Converting Existing Models

To convert existing CUDA-trained models to TPU-compatible format, use the provided conversion script:

```bash
python -m module.tpu.convert_model --cuda_checkpoint="path/to/cuda_model.pth" --tpu_output="path/to/tpu_model.pth"
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**: TPU requires static shapes for tensors. Ensure your model doesn't use dynamic shapes.

2. **Out of Memory Errors**: Add more `mark_step()` calls to your code to force execution of pending operations.

3. **Performance Issues**: Try using bfloat16 precision and ensure you're using TPU-optimized operations.

4. **Slow First Run**: The first execution on TPU includes compilation time. Subsequent runs will be faster.

### Getting Help

If you encounter issues with the TPU-optimized models, check the following resources:

- [PyTorch/XLA Documentation](https://pytorch.org/xla/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs/)
- [GitHub Issues](https://github.com/RVC-Boss/GPT-SoVITS/issues)