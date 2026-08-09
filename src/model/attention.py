import math
import torch
import torch.nn as nn


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, q_latent_dim, kv_latent_dim):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        head_dim = d_model // n_heads

        # Query projections
        self.wq_d = nn.Linear(d_model, q_latent_dim)

        # Precomputed matrix multiplication of w_k^U, for multiple heads
        self.w_qk = nn.Linear(q_latent_dim, n_heads * kv_latent_dim)

        # key/value latent projections
        self.wkv_d = nn.Linear(d_model, kv_latent_dim)
        self.wv_u = nn.Linear(kv_latent_dim, n_heads * head_dim)

        # Output projection
        self.wo = nn.Linear(n_heads * head_dim, d_model)

    def apply_rotary_emb(self, x, freqs_cis):
        """Apply rotary position embeddings to query/key projections.
        
        x: (batch, seq_len, n_heads, latent_dim)
        freqs_cis: (cos, sin) each (1, 1, seq_len, latent_dim)
        """
        cos, sin = freqs_cis
        # x shape: (B, T, H, D)
        # cos/sin shape: (1, 1, T, D) - broadcastable
        # Split into pairs for rotation
        x_real = x[..., 0::2]
        x_imag = x[..., 1::2]
        cos = cos[..., :x_real.shape[-1]]
        sin = sin[..., :x_imag.shape[-1]]
        # Apply rotation
        x_rotated_real = x_real * cos - x_imag * sin
        x_rotated_imag = x_real * sin + x_imag * cos
        # Interleave back
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_real
        x_rotated[..., 1::2] = x_rotated_imag
        return x_rotated

    def forward(self, x, freqs_cis=None):
        batch_size, seq_len, d_model = x.shape

        # Projection of input into latent space
        c_q = self.wq_d(x)  # (batch_size, seq_len, q_latent_dim)
        c_kv = self.wkv_d(x)  # (batch_size, seq_len, kv_latent_dim)

        # Project queries to heads
        q = self.w_qk(c_q).view(batch_size, seq_len, self.n_heads, self.kv_latent_dim)
        
        # Apply rotary embeddings to queries if provided
        if freqs_cis is not None:
            q = self.apply_rotary_emb(q, freqs_cis)

        # Attention scores: q @ k^T / sqrt(d)
        # q: (B, T, H, D), c_kv: (B, T, D_kv)
        # scores: (B, H, T, T)
        scores = torch.matmul(
            q.transpose(1, 2), 
            c_kv.transpose(-2, -1)[:, None, ...]
        ) / math.sqrt(self.kv_latent_dim)

        # attention computation
        attn_weight = torch.softmax(scores, dim=-1)

        # Restore V from latent space
        v = self.wv_u(c_kv).view(batch_size, seq_len, self.n_heads, -1)

        # Compute attention output, shape: (batch_size, seq_len, n_heads, head_dim)
        attn_output = torch.matmul(attn_weight, v.transpose(1, 2)).transpose(1, 2).contiguous()

        # Concatenate heads, then apply output projection
        attn_output = self.wo(attn_output.view(batch_size, seq_len, -1))
        
        return attn_output