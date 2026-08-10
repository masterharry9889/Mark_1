import torch
import torch.nn as nn
# parallel/pipeline_parallel.py
class PipelineParallel:
    """Split transformer layers across GPU groups."""
    def __init__(self, model, pp_size):
        self.pp_size = pp_size
        self.stages = self._partition_layers(model, pp_size)

    def _partition_layers(self, model, pp_size):
        layers_per_stage = len(model.blocks) // pp_size
        stages = []
        for i in range(pp_size):
            start = i * layers_per_stage
            end = start + layers_per_stage
            stages.append(model.blocks[start:end])
        return stages

    def forward_microbatch(self, x, stage_idx):
        """Run one microbatch through one pipeline stage."""
        return self.stages[stage_idx](x)

