import os
import torch
import argparse

# Import TPU utilities
from module.tpu.utils import get_xla_device

# Import original CUDA models
from module.models import SynthesizerTrn as CUDASynthesizerTrn

# Import TPU-optimized models
from module.tpu.models import SynthesizerTrn as TPUSynthesizerTrn

def convert_model(cuda_checkpoint_path, tpu_output_path, device=None):
    """
    Convert a CUDA-trained model checkpoint to a TPU-compatible model checkpoint.
    
    Args:
        cuda_checkpoint_path: Path to the CUDA model checkpoint
        tpu_output_path: Path to save the TPU-compatible model checkpoint
        device: Device to load the model on (default: XLA device if available, else CPU)
    """
    print(f"Converting model from {cuda_checkpoint_path} to {tpu_output_path}")
    
    # Determine device
    if device is None:
        try:
            device = get_xla_device()
            print(f"Using XLA device: {device}")
        except:
            device = torch.device('cpu')
            print("XLA not available, using CPU")
    
    # Load CUDA checkpoint
    print("Loading CUDA checkpoint...")
    cuda_checkpoint = torch.load(cuda_checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    model_config = cuda_checkpoint.get('config', {})
    
    # Create CUDA model with the same configuration
    cuda_model = CUDASynthesizerTrn(
        model_config.get('spec_channels', 80),
        model_config.get('segment_size', 8192),
        model_config.get('inter_channels', 192),
        model_config.get('hidden_channels', 192),
        model_config.get('filter_channels', 768),
        model_config.get('n_heads', 2),
        model_config.get('n_layers', 6),
        model_config.get('kernel_size', 3),
        model_config.get('p_dropout', 0.1),
        model_config.get('resblock', '1'),
        model_config.get('resblock_kernel_sizes', [3, 7, 11]),
        model_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        model_config.get('upsample_rates', [8, 8, 2, 2]),
        model_config.get('upsample_initial_channel', 512),
        model_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
        model_config.get('n_speakers', 0),
        model_config.get('gin_channels', 0),
        model_config.get('use_sdp', True),
    )
    
    # Load state dict into CUDA model
    cuda_model.load_state_dict(cuda_checkpoint['model'])
    
    # Create TPU model with the same configuration
    print("Creating TPU model...")
    tpu_model = TPUSynthesizerTrn(
        model_config.get('spec_channels', 80),
        model_config.get('segment_size', 8192),
        model_config.get('inter_channels', 192),
        model_config.get('hidden_channels', 192),
        model_config.get('filter_channels', 768),
        model_config.get('n_heads', 2),
        model_config.get('n_layers', 6),
        model_config.get('kernel_size', 3),
        model_config.get('p_dropout', 0.1),
        model_config.get('resblock', '1'),
        model_config.get('resblock_kernel_sizes', [3, 7, 11]),
        model_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        model_config.get('upsample_rates', [8, 8, 2, 2]),
        model_config.get('upsample_initial_channel', 512),
        model_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
        model_config.get('n_speakers', 0),
        model_config.get('gin_channels', 0),
        model_config.get('use_sdp', True),
    )
    
    # Transfer weights from CUDA model to TPU model
    tpu_model.load_state_dict(cuda_model.state_dict())
    
    # Create new checkpoint with TPU model
    tpu_checkpoint = {
        'model': tpu_model.state_dict(),
        'config': model_config,
        'iteration': cuda_checkpoint.get('iteration', 0),
        'optimizer': cuda_checkpoint.get('optimizer', None),
        'learning_rate': cuda_checkpoint.get('learning_rate', 0.0),
    }
    
    # Save TPU checkpoint
    print(f"Saving TPU checkpoint to {tpu_output_path}")
    os.makedirs(os.path.dirname(tpu_output_path), exist_ok=True)
    torch.save(tpu_checkpoint, tpu_output_path)
    
    print("Conversion complete!")
    return tpu_model

def main():
    parser = argparse.ArgumentParser(description='Convert CUDA model to TPU-compatible model')
    parser.add_argument('--cuda_checkpoint', type=str, required=True, help='Path to CUDA model checkpoint')
    parser.add_argument('--tpu_output', type=str, required=True, help='Path to save TPU-compatible model checkpoint')
    args = parser.parse_args()
    
    convert_model(args.cuda_checkpoint, args.tpu_output)

if __name__ == '__main__':
    main()