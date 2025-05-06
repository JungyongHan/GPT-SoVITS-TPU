import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import attentions
from module.mrte_model import MRTE
from module.tpu.utils import mark_step

# Import symbols for text processing
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        version="v2",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.version = version

        # Project SSL features to hidden dimension
        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        # Encoder for SSL features
        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        # Encoder for text features
        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        # Select appropriate symbol set based on version
        if self.version == "v1":
            symbols = symbols_v1.symbols
        else:
            symbols = symbols_v2.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        # Multi-reference Text Encoder
        self.mrte = MRTE()

        # Second encoder stage
        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        # Final projection layer
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge, speed=1, test=None):
        # Create mask for SSL features
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)

        # Process SSL features
        y = self.ssl_proj(y * y_mask) * y_mask
        
        # Mark step for TPU execution
        mark_step()
        
        y = self.encoder_ssl(y * y_mask, y_mask)
        
        # Mark step for TPU execution
        mark_step()

        # Create mask for text features
        text_mask = torch.unsqueeze(commons.sequence_mask(text_lengths, text.size(1)), 1).to(y.dtype)
        
        # Zero out text if test mode is enabled
        if test == 1:
            text = torch.zeros_like(text)
            
        # Process text features
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        
        # Mark step for TPU execution
        mark_step()
        
        # Apply MRTE
        y = self.mrte(y, y_mask, text, text_mask, ge)
        
        # Mark step for TPU execution
        mark_step()
        
        # Apply second encoder
        y = self.encoder2(y * y_mask, y_mask)
        
        # Handle speed adjustment using TPU-friendly interpolation
        if speed != 1:
            # Calculate new size statically when possible
            new_size = int(y.shape[-1] / speed) + 1
            y = F.interpolate(y, size=new_size, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
        
        # Final projection
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        return y, m, logs, y_mask

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        
        # Mark step for TPU execution
        mark_step()
        
        y = self.encoder_ssl(y * y_mask, y_mask)
        
        # Mark step for TPU execution
        mark_step()

        y = self.mrte(y, y_mask, refer, refer_mask, ge)
        
        # Mark step for TPU execution
        mark_step()

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized