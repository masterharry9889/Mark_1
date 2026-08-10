import torch

# training/scheduler.py
class WarmupStableDecay:
    """WSD schedule: warmup → stable → linear decay."""
    def __init__(self, optimizer, warmup_steps=2000, stable_steps=900000,
                 decay_steps=100000, max_lr=3e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr

    def step(self, step_num):
        if step_num < self.warmup_steps:
            lr = self.max_lr * step_num / self.warmup_steps
        elif step_num < self.warmup_steps + self.stable_steps:
            lr = self.max_lr
        else:
            decay_progress = (step_num - self.warmup_steps - self.stable_steps) / self.decay_steps
            lr = self.max_lr * (1 - decay_progress)

        for group in self.optimizer.param_groups:
            group['lr'] = lr