# TPU-Optimized Models for GPT-SoVITS

This directory contains TPU-optimized versions of the GPT-SoVITS models, designed to work efficiently with PyTorch/XLA 2.5.0 and above. These models have been adapted from the original CUDA-based implementations to leverage the performance benefits of TPU acceleration.

## Installation Requirements

To use these TPU-optimized models, you need to install PyTorch/XLA:

```bash
pip install torch==2.5.0
pip install torch_xla==2.5.0
```

Make sure you have access to a TPU device, either through Google Cloud TPU VMs or Colab TPU runtimes.

## Usage

Instead of importing models from the standard module path, import the TPU-optimized versions:

```python
# Replace this:
# from module.models import SynthesizerTrn, MultiPeriodDiscriminator

# With this:
from module.tpu.models import SynthesizerTrn
from module.tpu.discriminator import MultiPeriodDiscriminator
```

### TPU Device Setup

Before using the models, set up the TPU device:

```python
from module.tpu.utils import get_xla_device, mark_step

# Get the XLA device
device = get_xla_device()

# Move your model to the TPU device
model = SynthesizerTrn(...).to(device)
discriminator = MultiPeriodDiscriminator().to(device)
```

### Training Loop Example

When training on TPU, it's important to use the `mark_step()` function at appropriate points to optimize compilation and execution:

```python
from module.tpu.utils import mark_step

# Training loop
for batch in dataloader:
    # Move data to TPU
    x, x_lengths, y, y_lengths, sid = [item.to(device) for item in batch]
    
    # Forward pass
    y_hat, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model(x, x_lengths, y, y_lengths, sid)
    
    # Mark step after forward pass
    mark_step()
    
    # Discriminator forward
    y_d_hat_r, y_d_hat_g, _, _ = discriminator(y, y_hat)
    
    # Mark step after discriminator
    mark_step()
    
    # Calculate losses and update weights
    # ...
    
    # Mark step after backward pass
    mark_step()
```

## Key Optimizations

The TPU-optimized models include several key optimizations:

1. **TPU-Safe Random Number Generation**: Uses specialized functions for generating random numbers that work well with TPU compilation.

2. **Strategic Compilation Breaks**: Added `mark_step()` calls at appropriate points to optimize TPU compilation and execution.

3. **Static Shapes**: Modified operations to prefer static shapes where possible, which improves TPU performance.

4. **Memory-Efficient Operations**: Replaced certain operations with more TPU-friendly alternatives to reduce memory usage and improve performance.

5. **Optimized Data Movement**: Minimized unnecessary data transfers between host and device.

## Model Structure

The TPU-optimized models maintain the same architecture and functionality as the original models, but with TPU-specific optimizations. The following components have been optimized:

- `StochasticDurationPredictor`
- `DurationPredictor`
- `TextEncoder`
- `ResidualCouplingBlock`
- `PosteriorEncoder`
- `Generator`
- `SynthesizerTrn`
- `MultiPeriodDiscriminator`

## Troubleshooting

If you encounter compilation errors or performance issues:

1. Ensure you're using compatible versions of PyTorch and PyTorch/XLA
2. Check that your input tensors have consistent shapes across batches
3. Try adding additional `mark_step()` calls if you encounter OOM errors
4. For large models, consider using model parallelism or gradient checkpointing

## Performance Comparison

On supported TPU hardware, these optimized models can provide significant speedups compared to the original CUDA implementation, especially for larger batch sizes and sequence lengths.