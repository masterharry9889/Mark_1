import torch
import torch.nn as nn

# training/optimizer.py
class MoEAdamW:
    """Separate LR groups: router 10x higher."""
    def __init__(self, model, lr=3e-4, wd=0.1):
        router, expert, other = [], [], []
        for name, p in model.named_parameters():
            if 'router' in name or 'gate' in name:
                router.append(p)
            elif 'expert' in name:
                expert.append(p)
            else:
                other.append(p)

        self.optimizer = torch.optim.AdamW([
            {'params': router, 'lr': lr * 10},  # Router: fast adaptation
            {'params': expert, 'lr': lr},
            {'params': other, 'lr': lr},
        ], weight_decay=wd, betas=(0.9, 0.95))
    
    def __getattr__(self, name):
        return getattr(self.optimizer, name)
    
    def step(self):
        return self.optimizer.step()
    
    def zero_grad(self):
        return self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups