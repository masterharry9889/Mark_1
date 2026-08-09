# inference/quantize.py
class MoEQuantizer:
    """Per-expert quantization (INT8/INT4)."""
    def __init__(self, model):
        self.model = model

    def quantize_experts(self, method='awq', bits=4):
        """Quantize each expert independently."""
        for layer in self.model.blocks:
            for expert in layer.moe.experts:
                expert = self._quantize(expert, method, bits)

    def _quantize(self, module, method, bits):
        if method == 'awq':
            # AWQ: activation-aware weight quantization
            return self._awq_quantize(module, bits)
        elif method == 'gptq':
            return self._gptq_quantize(module, bits)
        return module