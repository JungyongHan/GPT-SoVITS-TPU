import torch
from torch import nn
from torch.nn import functional as F

from module.tpu.utils import mark_step

class MRTE(nn.Module):
    """TPU-optimized Multi-Reference Text Encoder"""
    def __init__(self):
        super().__init__()
        self.text_projection = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale = nn.Parameter(torch.ones(1))
        self.text_projection_scale.requires_grad = True
        
        self.text_projection_2 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_2 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_2.requires_grad = True
        
        self.text_projection_3 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_3 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_3.requires_grad = True
        
        self.text_projection_4 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_4 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_4.requires_grad = True
        
        self.text_projection_5 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_5 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_5.requires_grad = True
        
        self.text_projection_6 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_6 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_6.requires_grad = True
        
        self.text_projection_7 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_7 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_7.requires_grad = True
        
        self.text_projection_8 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_8 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_8.requires_grad = True
        
        self.text_projection_9 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_9 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_9.requires_grad = True
        
        self.text_projection_10 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_10 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_10.requires_grad = True
        
        self.text_projection_11 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_11 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_11.requires_grad = True
        
        self.text_projection_12 = nn.Conv1d(192, 768, kernel_size=1)
        self.text_projection_scale_12 = nn.Parameter(torch.ones(1))
        self.text_projection_scale_12.requires_grad = True

    def forward(self, y, y_mask, text, text_mask, ge):
        # TPU-optimized implementation of MRTE forward pass
        
        # Apply text projections with scaling
        text_1 = self.text_projection(text) * self.text_projection_scale
        text_2 = self.text_projection_2(text) * self.text_projection_scale_2
        text_3 = self.text_projection_3(text) * self.text_projection_scale_3
        text_4 = self.text_projection_4(text) * self.text_projection_scale_4
        
        # Mark step for TPU execution after first batch of projections
        mark_step()
        
        text_5 = self.text_projection_5(text) * self.text_projection_scale_5
        text_6 = self.text_projection_6(text) * self.text_projection_scale_6
        text_7 = self.text_projection_7(text) * self.text_projection_scale_7
        text_8 = self.text_projection_8(text) * self.text_projection_scale_8
        
        # Mark step for TPU execution after second batch of projections
        mark_step()
        
        text_9 = self.text_projection_9(text) * self.text_projection_scale_9
        text_10 = self.text_projection_10(text) * self.text_projection_scale_10
        text_11 = self.text_projection_11(text) * self.text_projection_scale_11
        text_12 = self.text_projection_12(text) * self.text_projection_scale_12
        
        # Mark step for TPU execution after third batch of projections
        mark_step()
        
        # Apply masks to text projections
        text_1 = text_1 * text_mask
        text_2 = text_2 * text_mask
        text_3 = text_3 * text_mask
        text_4 = text_4 * text_mask
        text_5 = text_5 * text_mask
        text_6 = text_6 * text_mask
        text_7 = text_7 * text_mask
        text_8 = text_8 * text_mask
        text_9 = text_9 * text_mask
        text_10 = text_10 * text_mask
        text_11 = text_11 * text_mask
        text_12 = text_12 * text_mask
        
        # Calculate attention scores with TPU-friendly operations
        # Using einsum for efficient matrix multiplication on TPU
        attn_1 = torch.einsum('bct,bcs->bts', y, text_1)
        attn_2 = torch.einsum('bct,bcs->bts', y, text_2)
        attn_3 = torch.einsum('bct,bcs->bts', y, text_3)
        
        # Mark step for TPU execution after first batch of attention calculations
        mark_step()
        
        attn_4 = torch.einsum('bct,bcs->bts', y, text_4)
        attn_5 = torch.einsum('bct,bcs->bts', y, text_5)
        attn_6 = torch.einsum('bct,bcs->bts', y, text_6)
        
        # Mark step for TPU execution after second batch of attention calculations
        mark_step()
        
        attn_7 = torch.einsum('bct,bcs->bts', y, text_7)
        attn_8 = torch.einsum('bct,bcs->bts', y, text_8)
        attn_9 = torch.einsum('bct,bcs->bts', y, text_9)
        
        # Mark step for TPU execution after third batch of attention calculations
        mark_step()
        
        attn_10 = torch.einsum('bct,bcs->bts', y, text_10)
        attn_11 = torch.einsum('bct,bcs->bts', y, text_11)
        attn_12 = torch.einsum('bct,bcs->bts', y, text_12)
        
        # Apply masks to attention scores
        attn_mask = torch.unsqueeze(y_mask, -1) * torch.unsqueeze(text_mask, 1)
        
        # Apply scaling and masking to attention scores
        attn_1 = attn_1 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_2 = attn_2 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_3 = attn_3 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_4 = attn_4 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        
        # Mark step for TPU execution after first batch of attention masking
        mark_step()
        
        attn_5 = attn_5 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_6 = attn_6 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_7 = attn_7 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_8 = attn_8 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        
        # Mark step for TPU execution after second batch of attention masking
        mark_step()
        
        attn_9 = attn_9 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_10 = attn_10 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_11 = attn_11 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        attn_12 = attn_12 * attn_mask / torch.sqrt(torch.tensor(768.0, device=y.device))
        
        # Apply softmax to attention scores
        attn_1 = F.softmax(attn_1, dim=2)
        attn_2 = F.softmax(attn_2, dim=2)
        attn_3 = F.softmax(attn_3, dim=2)
        attn_4 = F.softmax(attn_4, dim=2)
        attn_5 = F.softmax(attn_5, dim=2)
        attn_6 = F.softmax(attn_6, dim=2)
        
        # Mark step for TPU execution after first batch of softmax
        mark_step()
        
        attn_7 = F.softmax(attn_7, dim=2)
        attn_8 = F.softmax(attn_8, dim=2)
        attn_9 = F.softmax(attn_9, dim=2)
        attn_10 = F.softmax(attn_10, dim=2)
        attn_11 = F.softmax(attn_11, dim=2)
        attn_12 = F.softmax(attn_12, dim=2)
        
        # Calculate context vectors with TPU-friendly operations
        context_1 = torch.einsum('bts,bcs->bct', attn_1, text)
        context_2 = torch.einsum('bts,bcs->bct', attn_2, text)
        context_3 = torch.einsum('bts,bcs->bct', attn_3, text)
        
        # Mark step for TPU execution after first batch of context calculations
        mark_step()
        
        context_4 = torch.einsum('bts,bcs->bct', attn_4, text)
        context_5 = torch.einsum('bts,bcs->bct', attn_5, text)
        context_6 = torch.einsum('bts,bcs->bct', attn_6, text)
        
        # Mark step for TPU execution after second batch of context calculations
        mark_step()
        
        context_7 = torch.einsum('bts,bcs->bct', attn_7, text)
        context_8 = torch.einsum('bts,bcs->bct', attn_8, text)
        context_9 = torch.einsum('bts,bcs->bct', attn_9, text)
        
        # Mark step for TPU execution after third batch of context calculations
        mark_step()
        
        context_10 = torch.einsum('bts,bcs->bct', attn_10, text)
        context_11 = torch.einsum('bts,bcs->bct', attn_11, text)
        context_12 = torch.einsum('bts,bcs->bct', attn_12, text)
        
        # Combine context vectors
        context = context_1 + context_2 + context_3 + context_4 + context_5 + context_6 + context_7 + context_8 + context_9 + context_10 + context_11 + context_12
        
        # Add global embedding if provided
        if ge is not None:
            context = context + ge
        
        # Combine with original input
        y = y + context
        
        # Mark final step for TPU execution
        mark_step()
        
        return y