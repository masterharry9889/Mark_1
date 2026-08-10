import torch
import torch.nn.functional as F
# losses/cross_entropy.py
class FusedCrossEntropy:
    """Fused CE with optional label smoothing.
    Use flash-attn's fused CE if available, else fallback.
    """
    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        # logits: (B, T, V), labels: (B, T)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            label_smoothing=self.label_smoothing
        )
