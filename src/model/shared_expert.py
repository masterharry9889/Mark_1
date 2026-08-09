import torch.nn as nn

from .expert import SwiGLUExpert
class SharedExpert(nn.Module):
    """Always-active expert — processes every token.
    Provides stable baseline that routed experts refine.
    """
    def __init__(self, d_model, d_ff_shared):
        super().__init__()
        self.expert = SwiGLUExpert(d_model, d_ff_shared)
    
    def forward(self, x):
        return self.expert(x) 