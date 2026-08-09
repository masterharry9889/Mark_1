# parallel/tensor_parallel.py
class TensorParallel:
    """Split weight matrices across GPUs (column/row parallel)."""
    def __init__(self, module, tp_size):
        self.tp_size = tp_size
        self.rank = dist.get_rank()

    def column_parallel(self, weight, bias=None):
        """Split output dimension: weight shape (d_out, d_in) → (d_out/tp, d_in)"""
        return nn.Linear(
            weight.shape[1], weight.shape[0] // self.tp_size,
            bias=bias is not None
        )

    def row_parallel(self, weight, bias=None):
        """Split input dimension: weight shape (d_out, d_in) → (d_out, d_in/tp)"""
        return nn.Linear(
            weight.shape[1] // self.tp_size, weight.shape[0],
            bias=bias is not None
        )

