from flash_attn.layers.rotary import apply_rotary_emb

import torch
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding as _LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding as _LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding as _LlamaDynamicNTKScalingRotaryEmbedding,
)

class LlamaRotaryEmbedding(_LlamaRotaryEmbedding):

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        emb = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaLinearScalingRotaryEmbedding(_LlamaLinearScalingRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        emb = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(_LlamaDynamicNTKScalingRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        emb = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    assert unsqueeze_dim == 1, "fused rotary pos emb only supports unsqueeze_dim=1"
    assert q.shape[-1] == cos.shape[-1]*2, "q and cos must have the same embedding dimension"
    if len(position_ids.shape) == 2:
        assert position_ids.shape[0] == 1, "position_ids must have a batch dimension of 1"
        position_ids = position_ids[0]
    
    cos = cos[position_ids]
    sin = sin[position_ids]
    q_embed = apply_rotary_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
    k_embed = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
    return q_embed, k_embed