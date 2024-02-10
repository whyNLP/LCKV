# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import os
import math
import warnings
from tqdm import trange
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from transformers.models.llama.modeling_llama import (
    LlamaModel as _LlamaModel,
    LlamaForCausalLM as _LlamaForCausalLM,
    LlamaDecoderLayer as _LlamaDecoderLayer,
    LlamaAttention as _LlamaAttention,
    LlamaFlashAttention2 as _LlamaFlashAttention2,
    _prepare_4d_causal_attention_mask,
    LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
    LLAMA_INPUTS_DOCSTRING,
    logger
)

from .configuration_llama import OptLlamaConfig

def apply_rotary_pos_emb_q(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

class DummyContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

dummy_context = DummyContext()

class LlamaAttentionBase(_LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper
    It behaves exactly the same as its parent, we just add an input encoder_outputs."""

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

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        bsz, q_len, _ = hidden_states.size()
        if encoder_outputs is not None:
            kv_seq_len = encoder_outputs[0].shape[-2]
        elif past_key_value is not None:
            kv_seq_len = q_len + past_key_value[0].shape[-2]
        else:
            kv_seq_len = q_len

        if q_len == 1 and kv_seq_len == 1:
            forward_func = self._forward_dummy
        elif q_len == kv_seq_len:
            forward_func = self._forward_training
        elif q_len == 1:
            forward_func = self._forward_decoding
        else:
            raise ValueError(f"Invalid q_len: {q_len} and kv_seq_len: {kv_seq_len}")

        return forward_func(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            encoder_outputs,
            output_attentions,
            use_cache,
            **kwargs,
        )
    
    def _forward_dummy(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if use_cache:
            query_states, key_states, value_states, kv_seq_len, past_key_value = self._get_qkv(
                hidden_states,
                position_ids,
                past_key_value,
                encoder_outputs,
                use_cache,
                **kwargs
            )

        attn_output = torch.zeros_like(hidden_states)

        return attn_output, None, past_key_value

    def _forward_training(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states, kv_seq_len, past_key_value = self._get_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            encoder_outputs,
            use_cache,
            **kwargs
        )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        query_states = query_states[:, :, 1:, :]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len-1, kv_seq_len-1):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len-1, kv_seq_len-1)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask[:, :, :-1, :-1]

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = torch.cat([torch.zeros(bsz, self.num_heads, 1, self.head_dim, dtype=attn_output.dtype, device=attn_output.device), attn_output], dim=2)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def _forward_decoding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states, kv_seq_len, past_key_value = self._get_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            encoder_outputs,
            use_cache,
            **kwargs
        )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len-1):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len-1)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            raise ValueError(
                "Attention mask is not supported for decoding."
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class LlamaFlashAttention2Base(_LlamaFlashAttention2):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    It behaves exactly the same as its parent, we just add an input encoder_outputs.
    """

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

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()
        if encoder_outputs is not None:
            kv_seq_len = encoder_outputs[0].shape[-2]
        elif past_key_value is not None:
            kv_seq_len = q_len + past_key_value[0].shape[-2]
        else:
            kv_seq_len = q_len

        if q_len == 1 and kv_seq_len == 1:
            forward_func = self._forward_dummy
        elif q_len == kv_seq_len:
            forward_func = self._forward_training
        elif q_len == 1:
            forward_func = self._forward_decoding
        else:
            raise ValueError(f"Invalid q_len: {q_len} and kv_seq_len: {kv_seq_len}")

        return forward_func(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            encoder_outputs,
            output_attentions,
            use_cache,
            **kwargs,
        )
    
    def _forward_dummy(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if use_cache:
            query_states, key_states, value_states, kv_seq_len, past_key_value = self._get_qkv(
                hidden_states,
                position_ids,
                past_key_value,
                encoder_outputs,
                use_cache,
                **kwargs
            )

        attn_output = torch.zeros_like(hidden_states)

        return attn_output, None, past_key_value

    def _forward_training(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states, kv_seq_len, past_key_value = self._get_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            encoder_outputs,
            use_cache,
            **kwargs
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        query_states = query_states[:, 1:, :, :]

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len-1, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len-1, self.hidden_size)
        attn_output = torch.cat([torch.zeros(bsz, 1, self.hidden_size, dtype=attn_output.dtype, device=attn_output.device), attn_output], dim=1)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def _forward_decoding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states, kv_seq_len, past_key_value = self._get_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            encoder_outputs,
            use_cache,
            **kwargs
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaAttention(LlamaAttentionBase):
    def _get_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if encoder_outputs is not None:
            output = self._get_qkv_encoder(
                hidden_states,
                position_ids,
                None,
                encoder_outputs,
                use_cache,
                **kwargs
            )
        else:
            output = self._get_qkv_cache(
                hidden_states,
                position_ids,
                past_key_value,
                None,
                use_cache,
                **kwargs
            )
        return output
    
    def _get_qkv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        return super()._get_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            None,
            use_cache,
            **kwargs
        )

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

        if use_cache:
            _key_states = self.k_proj(hidden_states)
            _value_states = self.v_proj(hidden_states)
            _key_states = _key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            _value_states = _value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            _key_states = apply_rotary_pos_emb_q(_key_states, cos, sin, position_ids)
            past_key_value = (_key_states, _value_states)
        else:
            past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value


class LlamaFlashAttention2(LlamaFlashAttention2Base):
    def _get_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if encoder_outputs is not None:
            output = self._get_qkv_encoder(
                hidden_states,
                position_ids,
                None,
                encoder_outputs,
                use_cache,
                **kwargs
            )
        else:
            output = self._get_qkv_cache(
                hidden_states,
                position_ids,
                past_key_value,
                None,
                use_cache,
                **kwargs
            )
        return output
    
    def _get_qkv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        return super()._get_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            None,
            use_cache,
            **kwargs
        )

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

        if use_cache:
            _key_states = self.k_proj(hidden_states)
            _value_states = self.v_proj(hidden_states)
            _key_states = _key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            _value_states = _value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            _key_states = apply_rotary_pos_emb_q(_key_states, cos, sin, position_ids)
            past_key_value = (_key_states, _value_states)
        else:
            past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value


class LlamaAttentionMiddle(LlamaAttention):
    def __init__(self, config):
        """Remove the key value projection."""
        super(_LlamaAttention, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
    
    def _get_qkv_cache(
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
        else:
            key_states = value_states = torch.zeros(bsz, self.num_key_value_heads, q_len-1, self.head_dim, dtype=query_states.dtype, device=query_states.device)

        past_key_value = None

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

        past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value


class LlamaFlashAttention2Middle(LlamaFlashAttention2):
    def __init__(self, config):
        """Remove the key value projection."""
        super(_LlamaAttention, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
    
    def _get_qkv_cache(
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
        else:
            key_states = value_states = torch.zeros(bsz, self.num_key_value_heads, q_len-1, self.head_dim, dtype=query_states.dtype, device=query_states.device)

        past_key_value = None

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

        past_key_value = None
        
        # remove the last token
        key_states = key_states[:, :, :-1, :]
        value_states = value_states[:, :, :-1, :]

        return query_states, key_states, value_states, kv_seq_len, past_key_value


class LlamaDecoderLayer(_LlamaDecoderLayer):
    def __init__(self, config: OptLlamaConfig, layer_idx: int):
        super(_LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        attn_cls = self._get_attn_cls(config, layer_idx)
        self.self_attn = attn_cls(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def _get_attn_cls(self, config: OptLlamaConfig, layer_idx: int):
        layer_types = [int(x) for x in config.layer_types.split("_")]
        layer_type = layer_types[layer_idx]

        if not getattr(config, "_flash_attn_2_enabled", False):
            if layer_type == 0:
                return LlamaAttentionBase
            elif layer_type == 1:
                return LlamaAttentionMiddle
            elif layer_type == 2:
                return LlamaAttention
            else:
                raise ValueError(f"Unknwon layer type: {layer_type}")
        else:
            if layer_type == 0:
                return LlamaFlashAttention2Base
            elif layer_type == 1:
                return LlamaFlashAttention2Middle
            elif layer_type == 2:
                return LlamaFlashAttention2
            else:
                raise ValueError(f"Unknwon layer type: {layer_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel(_LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: OptLlamaConfig
    """
    config_class = OptLlamaConfig

    def __init__(self, config: OptLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            if len(past_key_values[0]) == 2:
                past_key_values_length = past_key_values[0][0].shape[2]
            else:
                past_key_values_length = past_key_values[0][0].shape[1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if use_cache in ("target", "target-only"):
                _use_cache = bool(idx == self.config.target_layer % self.config.num_hidden_layers)
            else:
                _use_cache = use_cache

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    encoder_outputs,
                    output_attentions,
                    _use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    encoder_outputs=encoder_outputs,
                    output_attentions=output_attentions,
                    use_cache=_use_cache,
                )

            hidden_states = layer_outputs[0]

            if _use_cache and use_cache == "target-only":
                return layer_outputs[2 if output_attentions else 1]

            if _use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            elif use_cache == "target":
                next_decoder_cache += (None,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(_LlamaForCausalLM):
    config_class = OptLlamaConfig

    def __init__(self, config):
        super(_LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training:
            # training
            return self.forward_training(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict
            )
        elif labels is not None:
            # inference
            if os.environ.get("ALGPT_INFERENCE", False):
                func = self.forward_inference
            else:
                func = self.forward_training
            return func(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict
            )
        else:
            # prediction
            return self.forward_predict(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict
            )

    def forward_training(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        assert past_key_values is None, "past_key_values is not supported for training."
        assert not use_cache, "use_cache is not supported for training."

        # initialize kv w/ zero
        bsz, q_len = input_ids.size()
        zero_states = torch.zeros(bsz, self.config.num_key_value_heads, q_len, self.config.hidden_size // self.config.num_attention_heads, device=input_ids.device, dtype=self.dtype)
        encoder_outputs = (zero_states, zero_states)
        
        for i in range(self.config.num_encoders):
            
            context = torch.no_grad() if i < self.config.num_encoders - self.config.num_trained_encoders else dummy_context

            with context:
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                encoder_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    encoder_outputs=encoder_outputs,
                    inputs_embeds=inputs_embeds,
                    use_cache="target-only", # we are using past_key_values to do decoding
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True, # we want to retrive the past_key_values
                )
            
            # if "old_key_states" not in locals():
            #     old_key_states = encoder_outputs[0]
            #     old_value_states = encoder_outputs[1]
            # else:
            #     print(i, F.mse_loss(old_key_states, encoder_outputs[0])+F.mse_loss(old_value_states, encoder_outputs[1]))
            #     old_key_states = encoder_outputs[0]
            #     old_value_states = encoder_outputs[1]
            # breakpoint()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            use_cache="target" if self.config.train_kv else False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if self.config.train_kv:
            # the loss to mimic KV and final hidden
            gold_key_state, gold_value_state = encoder_outputs
            pred_key_state, pred_value_state = outputs[1][self.config.target_layer]
            loss_kv = F.mse_loss(pred_key_state, gold_key_state) + F.mse_loss(pred_value_state, gold_value_state)

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if self.config.train_kv:
                loss = loss + loss_kv

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def forward_inference(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """This is extremely slow, only use it for the final testing."""
        
        seq_len = input_ids.shape[1]
        logits = []
        
        # since it is too slow, we'll use tqdm by default.
        for i in trange(seq_len, leave=False):
            m_input_ids = input_ids[:, i:i+1]
            m_attention_mask = attention_mask[:, :i+1] if attention_mask is not None else None
            m_position_ids = position_ids[:, i:i+1] if position_ids is not None else None
            m_inputs_embeds = inputs_embeds[:, i:i+1] if inputs_embeds is not None else None
            
            outputs = self.forward_predict_one(
                input_ids=m_input_ids,
                attention_mask=m_attention_mask,
                position_ids=m_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=m_inputs_embeds,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            logits.append(outputs.logits)
            past_key_values = outputs.past_key_values

        logits = torch.cat(logits, dim=1)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def forward_predict(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        seq_len = input_ids.shape[1]
        
        if seq_len > self.config.num_encoders+1:
            # long prompts use encoders to mimic the key value
            outputs = self.forward_predict_prompt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif seq_len > 1:
            # short prompts decode token by token
            logits = []
            for i in range(seq_len):
                m_input_ids = input_ids[:, i:i+1]
                m_attention_mask = attention_mask[:, :i+1] if attention_mask is not None else None
                m_position_ids = position_ids[:, i:i+1] if position_ids is not None else None
                m_inputs_embeds = inputs_embeds[:, i:i+1] if inputs_embeds is not None else None
                
                outputs = self.forward_predict_one(
                    input_ids=m_input_ids,
                    attention_mask=m_attention_mask,
                    position_ids=m_position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=m_inputs_embeds,
                    labels=None,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                logits.append(outputs.logits)
                past_key_values = outputs.past_key_values
            logits = torch.cat(logits, dim=1)

            if not return_dict:
                outputs = (None, logits, past_key_values)
            else:
                outputs = CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        
        else:
            # token generation
            outputs = self.forward_predict_one(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return outputs
    
    def forward_predict_prompt(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # initialize kv w/ zero
        bsz, q_len = input_ids.size()
        zero_states = torch.zeros(bsz, self.config.num_key_value_heads, q_len, self.config.hidden_size // self.config.num_attention_heads, device=input_ids.device, dtype=self.dtype)
        encoder_outputs = (zero_states, zero_states)
        
        for i in range(self.config.num_encoders):
            
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            encoder_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                use_cache="target-only", # we are using past_key_values to do decoding
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True, # we want to retrive the past_key_values
            )
            
            # if "old_key_states" not in locals():
            #     old_key_states = encoder_outputs[0]
            # else:
            #     print(i, F.mse_loss(old_key_states, encoder_outputs[0]))
            #     old_key_states = encoder_outputs[0]
            # breakpoint()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # manually set the key value
        if use_cache:
            layer_types = [int(x) for x in self.config.layer_types.split("_")]
            memory = outputs[1][self.config.target_layer]
            new_past_key_values = tuple(
                outputs[1][idx] if tp == 0 else memory
                for idx, tp in enumerate(layer_types)
            )
            if return_dict:
                outputs.past_key_values = new_past_key_values
            else:
                outputs = tuple(outputs[0], new_past_key_values, *outputs[2:])

        hidden_states = outputs[0]
        if os.environ.get("ALGPT_GENERATION", False):
            # only use the last token
            logits = self.lm_head(hidden_states[:,-1:,:])
        else:
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            raise NotImplementedError("labels is not supported for prompt generation.")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    
    def forward_predict_one(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # manually set the key value
        if use_cache:
            layer_types = [int(x) for x in self.config.layer_types.split("_")]
            memory = outputs[1][self.config.target_layer]
            new_past_key_values = tuple(
                outputs[1][idx] if tp == 0 else memory
                for idx, tp in enumerate(layer_types)
            )
            if return_dict:
                outputs.past_key_values = new_past_key_values
            else:
                outputs = tuple(outputs[0], new_past_key_values, *outputs[2:])

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            raise NotImplementedError("labels is not supported for token generation.")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
