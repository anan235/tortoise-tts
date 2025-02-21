# mla_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAAttention(nn.Module):
    def __init__(self, d_model, n_heads, latent_dim):
        """
        Multi-head Latent Attention (MLA) reduces the size of the key-value cache by
        projecting keys and values into a lower-dimensional latent space.
        
        Args:
            d_model (int): The model dimension.
            n_heads (int): Number of attention heads.
            latent_dim (int): The dimension for the latent projection.
        """
        super(MLAAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.latent_dim = latent_dim

        # Query projection remains standard.
        self.query_proj = nn.Linear(d_model, d_model)
        # Instead of direct key/value, project into a latent space.
        self.key_down_proj = nn.Linear(d_model, latent_dim)
        self.value_down_proj = nn.Linear(d_model, latent_dim)
        # Reconstruct full keys/values from latent representations.
        self.key_up_proj = nn.Linear(latent_dim, n_heads * self.head_dim)
        self.value_up_proj = nn.Linear(latent_dim, n_heads * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: Input tensor of shape (batch, seq, d_model).
            kv_cache: A dict containing 'K_latent' and 'V_latent' from previous tokens.
        Returns:
            output: The attention output.
            new_kv_cache: Updated KV cache containing the latent representations.
        """
        batch, seq, _ = x.shape
        # Compute query.
        Q = self.query_proj(x)  # (batch, seq, d_model)
        # Compute latent keys and values.
        K_latent = self.key_down_proj(x)  # (batch, seq, latent_dim)
        V_latent = self.value_down_proj(x)  # (batch, seq, latent_dim)

        # If previous cache exists, concatenate along the sequence dimension.
        if kv_cache is not None:
            K_latent = torch.cat([kv_cache['K_latent'], K_latent], dim=1)
            V_latent = torch.cat([kv_cache['V_latent'], V_latent], dim=1)
        new_kv_cache = {'K_latent': K_latent, 'V_latent': V_latent}

        # Reconstruct full keys/values.
        K = self.key_up_proj(K_latent)  # (batch, seq_total, n_heads * head_dim)
        V = self.value_up_proj(V_latent)  # (batch, seq_total, n_heads * head_dim)

        # Reshape for multi-head attention.
        seq_total = K.size(1)
        K = K.view(batch, seq_total, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_total, head_dim)
        V = V.view(batch, seq_total, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_total, head_dim)
        Q = Q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)         # (batch, n_heads, seq, head_dim)

        # Scaled dot-product attention.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (batch, n_heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        output = self.out_proj(context)
        return output, new_kv_cache
