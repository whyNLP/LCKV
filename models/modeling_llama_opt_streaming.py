import os
import math
import warnings
import types
from typing import List, Optional, Tuple, Union

import torch

from .modeling_llama_opt import (
    LlamaAttentionBase,
    LlamaAttention,
    LlamaAttentionMiddle,
    LlamaFlashAttention2Base,
    LlamaFlashAttention2,
    LlamaFlashAttention2Middle,
    apply_rotary_pos_emb_q
)

def _get_qkv(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    encoder_outputs: Optional[List[torch.FloatTensor]] = None,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb_q(key_states, cos, sin, key_position_ids)

    # remove the last token
    key_states = key_states[:, :, :-1, :]
    value_states = value_states[:, :, :-1, :]

    return query_states, key_states, value_states, kv_seq_len, past_key_value

def _get_qkv_encoder(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """It will deal with encoder_outputs differently from its parent."""
        assert past_key_value is None
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states, value_states = encoder_outputs

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_q(key_states, cos, sin, key_position_ids)

        if use_cache:
            _key_states = self.k_proj(hidden_states)
            _value_states = self.v_proj(hidden_states)
            _key_states = _key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            _value_states = _value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            past_key_value = (_key_states, _value_states)
        else:
            past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value

def _get_qkv_middle(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states, value_states = past_key_value
            key_position_ids = torch.arange(kv_seq_len - q_len, device=position_ids.device).unsqueeze(0)
            key_states = apply_rotary_pos_emb_q(key_states, cos, sin, key_position_ids)
        else:
            key_states = value_states = torch.zeros(bsz, self.num_key_value_heads, q_len-1, self.head_dim, dtype=query_states.dtype, device=query_states.device)

        past_key_value = None

        return query_states, key_states, value_states, kv_seq_len, past_key_value

def _get_qkv_middle_encoder(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """It will deal with encoder_outputs differently from its parent."""
        assert past_key_value is None
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states, value_states = encoder_outputs

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_q(key_states, cos, sin, key_position_ids)

        past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value

    
def _get_qkv_encoder_and_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Combine the kv from cache and encoder outputs"""
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states, value_states = encoder_outputs

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_q(key_states, cos, sin, key_position_ids)
        
        if use_cache:
            _key_states = self.k_proj(hidden_states)
            _value_states = self.v_proj(hidden_states)
            _key_states = _key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            _value_states = _value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            past_key_value = (_key_states, _value_states)
        else:
            past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value


def _get_qkv_middle_encoder_and_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Combine the kv from cache and encoder outputs"""
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states, value_states = encoder_outputs

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb_q(query_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_q(key_states, cos, sin, key_position_ids)
        
        past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value

def enable_opt_llama_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_opt_llama_pos_shift_attention(
                module,
            )

        if isinstance(module, (LlamaAttentionMiddle, LlamaFlashAttention2Middle)):
            model._modules[name]._get_qkv_cache = types.MethodType(
                _get_qkv_middle, model._modules[name]
            )
            model._modules[name]._get_qkv_encoder = types.MethodType(
                _get_qkv_middle_encoder, model._modules[name]
            )
            model._modules[name]._get_qkv_encoder_and_cache = types.MethodType(
                _get_qkv_middle_encoder_and_cache, model._modules[name]
            )
        elif isinstance(module, (LlamaAttention, LlamaFlashAttention2)):
            model._modules[name]._get_qkv_cache = types.MethodType(
                _get_qkv, model._modules[name]
            )
            model._modules[name]._get_qkv_encoder = types.MethodType(
                _get_qkv_encoder, model._modules[name]
            )
            model._modules[name]._get_qkv_encoder_and_cache = types.MethodType(
                _get_qkv_encoder_and_cache, model._modules[name]
            )
        elif isinstance(module, (LlamaAttentionBase, LlamaFlashAttention2Base)):
            model._modules[name]._get_qkv = types.MethodType(
                _get_qkv, model._modules[name]
            )

from streaming_llm.kv_cache import StartRecentKVCache

class OptStartRecentKVCache(StartRecentKVCache):
    """In our implementation, most of the KV caches are reference pointers.
    The original implementation of streammingLLM will copy the tensors.
    """
    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        # aviod duplicate copy
        deduplicated_key_values = []
        indices = []
        for item in past_key_values:
            for idx, ref in enumerate(deduplicated_key_values):
                if item is ref:
                    indices.append(idx)
                    break
            else:
                indices.append(len(deduplicated_key_values))
                deduplicated_key_values.append(item)
        
        # convert tensors
        deduplicated_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in deduplicated_key_values
        ]

        # restore references
        return [
            deduplicated_key_values[idx]
            for idx in indices
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        
        # aviod duplicate copy
        deduplicated_key_values = []
        indices = []
        for item in past_key_values:
            for idx, ref in enumerate(deduplicated_key_values):
                if item is ref:
                    indices.append(idx)
                    break
            else:
                indices.append(len(deduplicated_key_values))
                deduplicated_key_values.append(item)
        
        # convert tensors
        deduplicated_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in deduplicated_key_values
        ]

        # restore references
        return [
            deduplicated_key_values[idx]
            for idx in indices
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len

        # aviod duplicate copy
        deduplicated_key_values = []
        indices = []
        for item in past_key_values:
            for idx, ref in enumerate(deduplicated_key_values):
                if item is ref:
                    indices.append(idx)
                    break
            else:
                indices.append(len(deduplicated_key_values))
                deduplicated_key_values.append(item)
        
        # convert tensors
        deduplicated_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in deduplicated_key_values
        ]

        # restore references
        return [
            deduplicated_key_values[idx]
            for idx in indices
        ]


def enable_streaming_llm(model, start_size, recent_size):
    if "opt-llama" in model.config_class.model_type:
        k_seq_dim = v_seq_dim = 2

        enable_opt_llama_pos_shift_attention(model)
    elif "llama" in model.config_class.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config_class.model_type}")
    kv_cache = OptStartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache

