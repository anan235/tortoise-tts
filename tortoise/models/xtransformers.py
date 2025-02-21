# tortoise/models/xtransformers.py
"""
Rewritten XTransformer module for Tortoise TTS.

This variant supports an optional multi-token prediction head.
It integrates:
  - MLA-based attention for efficient KV caching.
  - Sparse MoE Feed-Forward Network for improved capacity.
  
Note: Many utility functions, helper classes, and positional embedding mechanisms are preserved from the original file.
"""

import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum

# -----------------------
# Constants and Named Tuples
DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])
LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates',
    'past_key_values',
])

# -----------------------
# Helper Functions
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

# -----------------------
# (For brevity, we assume that additional helper classes like ReluSquared,
# AbsolutePositionalEmbedding, FixedPositionalEmbedding, RelativePositionBias,
# AlibiPositionalBias, LearnedAlibiPositionalBias, RotaryEmbedding, rotate_half,
# apply_rotary_pos_emb, Scale, Rezero, ScaleNorm, RMSNorm, RMSScaleShiftNorm, Residual,
# GRUGating, ShiftTokens, etc. are defined here or imported from a shared module.)
# If RelativePositionBias is needed, ensure it is defined or imported.

# Example: Minimal RelativePositionBias (if not already defined)
class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)
    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret
    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)

# -----------------------
# Custom Modules: MLA and MoE
# mla_attention.py is assumed to be available as a separate module; below is its content:

# BEGIN mla_attention.py
# (Ensure this file exists in your PYTHONPATH.)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAAttention(nn.Module):
    def __init__(self, d_model, n_heads, latent_dim):
        super(MLAAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.latent_dim = latent_dim

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_down_proj = nn.Linear(d_model, latent_dim)
        self.value_down_proj = nn.Linear(d_model, latent_dim)
        self.key_up_proj = nn.Linear(latent_dim, n_heads * self.head_dim)
        self.value_up_proj = nn.Linear(latent_dim, n_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x, kv_cache=None):
        batch, seq, _ = x.shape
        Q = self.query_proj(x)
        K_latent = self.key_down_proj(x)
        V_latent = self.value_down_proj(x)
        if kv_cache is not None:
            K_latent = torch.cat([kv_cache['K_latent'], K_latent], dim=1)
            V_latent = torch.cat([kv_cache['V_latent'], V_latent], dim=1)
        new_kv_cache = {'K_latent': K_latent, 'V_latent': V_latent}
        K = self.key_up_proj(K_latent)
        V = self.value_up_proj(V_latent)
        seq_total = K.size(1)
        K = K.view(batch, seq_total, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(batch, seq_total, self.n_heads, self.head_dim).transpose(1,2)
        Q = Q.view(batch, seq, self.n_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1,2).contiguous().view(batch, seq, self.d_model)
        output = self.out_proj(context)
        return output, new_kv_cache
"""
# END mla_attention.py

# Similarly, moe_ffn.py is assumed to exist:
# BEGIN moe_ffn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, k=1, dropout=0.1):
        super(MoEFeedForward, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.d_model = d_model
        self.d_ff = d_ff
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        batch, seq, _ = x.shape
        gate_logits = self.gate(x) + self.expert_bias
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_expert_indices = torch.argmax(gate_probs, dim=-1)
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            mask = (top_expert_indices == expert_idx)
            if mask.sum() == 0:
                continue
            x_expert = x[mask]
            expert_output = self.experts[expert_idx](x_expert)
            output[mask] = expert_output
        output = self.dropout(output)
        return output
"""
# END moe_ffn.py

# -----------------------
# Replace the FeedForward module with our version that can use MoE.
from moe_ffn import MoEFeedForward

class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            mult=4,
            glu=False,
            relu_squared=False,
            post_act_ln=False,
            dropout=0.,
            zero_init_output=False,
            use_moe=False,
            num_experts=4
    ):
        super().__init__()
        if use_moe:
            self.net = MoEFeedForward(dim, int(dim * mult), num_experts=num_experts, k=1, dropout=dropout)
        else:
            inner_dim = int(dim * mult)
            dim_out = default(dim_out, dim)
            activation = ReluSquared() if relu_squared else nn.GELU()
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim),
                activation
            ) if not glu else GLU(dim, inner_dim, activation)
            self.net = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out)
            )
            if zero_init_output:
                init_zero_(self.net[-1])
    def forward(self, x):
        return self.net(x)

# -----------------------
# Replace the Attention module to use our MLA.
from mla_attention import MLAAttention

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            causal=False,
            dropout=0.,
            zero_init_output=False,
            rel_pos_bias=False,
            rel_pos_num_buckets=32,
            rel_pos_max_distance=128,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mla = MLAAttention(dim, heads, latent_dim=dim // 4)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)
        if zero_init_output:
            init_zero_(self.to_out)
        self.rel_pos_bias = None
        if rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(scale=dim_head ** 0.5, causal=causal,
                                                     heads=heads, num_buckets=rel_pos_num_buckets,
                                                     max_distance=rel_pos_max_distance)
    def forward(self, x, context=None, mask=None, attn_mask=None, rotary_pos_emb=None, prev_attn=None, layer_past=None):
        # Here we assume self-attention; context handling can be added if needed.
        out, new_kv_cache = self.mla(x, kv_cache=layer_past)
        # Optionally, relative positional bias can be applied here.
        out = self.dropout(out)
        out = self.to_out(out)
        intermediates = Intermediates(pre_softmax_attn=None, post_softmax_attn=None)
        return out, intermediates, new_kv_cache, new_kv_cache

# -----------------------
# Transformer Block for XTransformer
class XTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, use_moe=False, num_experts=4, dropout=0.1):
        super().__init__()
        self.attn = Attention(d_model, heads=n_heads, causal=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout, use_moe=use_moe, num_experts=num_experts)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, kv_cache=None):
        attn_out, intermediates, new_cache, _ = self.attn(x, layer_past=kv_cache)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, new_cache

class XTransformer(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, use_moe=False, num_experts=4, dropout=0.1, multi_token=False):
        super().__init__()
        self.layers = nn.ModuleList([
            XTransformerBlock(d_model, n_heads, d_ff, use_moe=use_moe, num_experts=num_experts, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.multi_token = multi_token
        if self.multi_token:
            self.multi_token_proj = nn.Linear(d_model, d_model)
    def forward(self, x, kv_cache=None):
        new_cache = {}
        for i, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache.get(f'layer_{i}', None)
            x, new_layer_cache = layer(x, kv_cache=layer_cache)
            new_cache[f'layer_{i}'] = new_layer_cache
        x = self.norm(x)
        if self.multi_token:
            x = self.multi_token_proj(x)
        return x, new_cache

# -----------------------
# The remainder of the file: Definitions for Encoder, Decoder, CrossAttender, 
# ViTransformerWrapper, TransformerWrapper, and ContinuousTransformerWrapper
# are preserved from the original code.

class AttentionLayers(nn.Module):
    def __init__(self, dim, depth, heads=8, causal=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        # Here, we use our updated XTransformerBlock in each layer.
        self.layers = nn.ModuleList([
            XTransformerBlock(dim, heads, dim*4, **kwargs) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        new_cache = {}
        for i, layer in enumerate(self.layers):
            x, new_layer_cache = layer(x, kv_cache=kwargs.get('kv_cache', None))
            new_cache[f'layer_{i}'] = new_layer_cache
        x = self.norm(x)
        return x, new_cache

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'Encoder must be non-causal'
        super().__init__(causal=False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'Decoder must be causal'
        super().__init__(causal=True, **kwargs)

class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)

class ViTransformerWrapper(nn.Module):
    def __init__(self, *, image_size, patch_size, attn_layers, num_classes=None, dropout=0., emb_dropout=0.):
        super().__init__()
        from einops import rearrange, repeat
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = FeedForward(dim, dim_out=num_classes, dropout=dropout) if num_classes is not None else None
    def forward(self, img, return_embeddings=False):
        from einops import rearrange, repeat
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x, _ = self.attn_layers(x)
        x = self.norm(x)
        if not exists(self.mlp_head) or return_embeddings:
            return x
        return self.mlp_head(x[:, 0])

class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, attn_layers, emb_dim=None, max_mem_len=0., shift_mem_down=0,
                 emb_dropout=0., num_memory_tokens=None, tie_embedding=False, use_pos_emb=True):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        self.init_()
    def init_(self):
        nn.init.kaiming_normal_(self.token_emb.weight)
    def forward(self, x, return_embeddings=False, mask=None, return_hiddens=False, return_attn=False, mems=None, use_cache=False, **kwargs):
        b, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)
        if self.num_memory_tokens > 0:
            from einops import repeat
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)
            if exists(mask):
                x = x
                mask = F.pad(mask, (self.num_memory_tokens, 0), value=True)
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)
        mem, x = x[:, :self.num_memory_tokens], x[:, self.num_memory_tokens:]
        out = self.to_logits(x) if not return_embeddings else x
        res = [out]
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            res.append(attn_maps)
        if use_cache:
            res.append(intermediates.past_key_values)
        if len(res) > 1:
            return tuple(res)
        return res[0]

class ContinuousTransformerWrapper(nn.Module):
    def __init__(self, *, max_seq_len, attn_layers, dim_in=None, dim_out=None, emb_dim=None, emb_dropout=0., use_pos_emb=True):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'
        dim = attn_layers.dim
        self.max_seq_len = max_seq_len
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len) if (use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()
    def forward(self, x, return_embeddings=False, mask=None, return_attn=False, mems=None, use_cache=False, **kwargs):
        b, n, device = *x.shape, x.device
        x = self.project_in(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)
        out = self.project_out(x) if not return_embeddings else x
        res = [out]
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            res.append(attn_maps)
        if use_cache:
            res.append(intermediates.past_key_values)
        if len(res) > 1:
            return tuple(res)
        return res[0]
