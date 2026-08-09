import torch
import math

# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if min == max:
        # Return all zeros (or all ones) when range is singular
        return torch.zeros(dim, dtype=torch.float32)

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0
class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        self._build_cache(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _build_cache(self, max_seq_len, device, dtype):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # emb shape: [seq_len, dim] - matches head_size = dim
        emb = torch.cat((freqs, freqs), dim=-1)

        mscale = self.mscale if isinstance(self.mscale, torch.Tensor) else torch.tensor(self.mscale, device=device, dtype=dtype)
        # Shape: (1, seq_len, 1, dim) for broadcasting with (B, T, H, D)
        self.register_buffer("cos_cached", (emb.cos() * mscale)[None, :, None, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * mscale)[None, :, None, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size] where head_size == self.dim
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Ensure inv_freq is on the same device as x
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
            self.mscale = self.mscale.to(x.device) if isinstance(self.mscale, torch.Tensor) else torch.tensor(self.mscale, device=x.device)
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)

        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. 
        # Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            # For extrapolation beyond original max, we should recompute inv_freq with new max_seq_len
            # but for now just extend cache with current inv_freq (which was computed for original_max_position_embeddings)
            self.max_seq_len_cached = seq_len
            self._build_cache(self.max_seq_len_cached, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :seq_len, ...].to(dtype=x.dtype),
        )

    def yarn(self, device=None):
        """Compute YaRN frequencies. Idempotent - can be called multiple times."""
        if device is None:
            device = self.inv_freq.device if hasattr(self, 'inv_freq') and self.inv_freq is not None else 'cpu'
        
        # Standard RoPE: theta_i = base^(-2i/dim) for i in [0, dim/2)
        # pos_freqs = base^(2i/dim), then inv_freq = 1/pos_freqs = base^(-2i/dim)
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) * 2.0 / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs  # High frequency (short wavelength) - for extrapolation
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)  # Low frequency (long wavelength) - for interpolation

        low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        
        # linear_ramp_mask returns 1 for low freq (interpolation), 0 for high freq (extrapolation)
        # We want: low freq -> use interpolation, high freq -> use extrapolation
        inv_freq_mask = linear_ramp_mask(low, high, self.dim // 2).float().to(device)
        
        # YaRN: extrapolate high frequencies, interpolate low frequencies
        # extrapolation_factor scales the extrapolated (high freq) dimensions
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask * self.extrapolation_factor

        self.register_buffer("inv_freq", inv_freq)
        
        mscale_val = get_mscale(self.scale) * self.attn_factor
        self.register_buffer("mscale", torch.tensor(mscale_val, device=device, dtype=torch.float32))

    def reset_parameters(self):
        """Reinitialize the embedding frequencies."""
        self._build_cache(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.get_default_dtype())