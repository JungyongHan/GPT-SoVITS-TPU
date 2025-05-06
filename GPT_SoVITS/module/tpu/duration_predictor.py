import torch
from torch import nn
from torch.nn import functional as F

from module import modules
from module.tpu.utils import mark_step

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        # Detach inputs to avoid unnecessary gradient computation
        x = torch.detach(x)
        
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
            
        # Apply first convolution with mask
        x = self.conv_1(x * x_mask)
        # Use ReLU activation which is well-supported on TPU
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        
        # Mark step for TPU execution after first block
        mark_step()
        
        # Apply second convolution with mask
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        
        # Mark step for TPU execution after second block
        mark_step()
        
        # Final projection
        x = self.proj(x * x_mask)
        return x * x_mask