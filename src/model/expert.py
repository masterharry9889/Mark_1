import torch
import torch.nn as nn
import torch.nn.functional as F
class SwiGLUExpert(nn.Module):
    """Single SwiGLU feed-forward expert.
    
    Args:
        d_model (int):  Hidden dimension (6144)
        d_ff (int):     FFN inner dimension (16384)
    """
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down  = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model) → output: (batch, seq_len, d_model)"""
        gate = self.w_gate(x)
        gate = gate * F.sigmoid(gate)  # SwiGLU gate
        up = self.w_up(x)  # up projection (should be on x, not gate)
        hidden = gate * up # Gate activation
        return self.w_down(hidden) # Down projection