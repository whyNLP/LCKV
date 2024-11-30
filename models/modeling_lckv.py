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
import copy
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    _prepare_4d_causal_attention_mask_with_cache_position,
    logger,
    repeat_kv,
    rotate_half,
)
from transformers.utils import add_start_docstrings_to_model_forward, is_flash_attn_greater_or_equal_2_10

from .cache_utils import AutoLayerCache, LayerCache
from .configuration_lckv import LCKVLlamaConfig
from .utils import IterStep, LayerTypeParser, flash_attention_forward


def apply_rotary(q, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class LCKVLlamaAttention(LlamaAttention):
    """
    LCKV Attention may not need to initialize weights for the key and value projections.
    """

    def __init__(self, config: LCKVLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.layer_type = LayerTypeParser(config.layer_types)[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type.use_sliding_window else None

        # Some layers may not need to compute key-value pairs
        if not self.layer_type.computes_kv:
            del self.k_proj
            del self.v_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = apply_rotary(query_states, cos, sin)

        # compute key and value states
        if self.layer_type.computes_kv:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_states = apply_rotary(key_states, cos, sin)

            if isinstance(past_key_value, Cache):
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            past_key_value.layer_set(self.layer_idx, key_states, value_states)

        # get the cached key and value states
        # if the layer attends to the top layers, there are two cases:
        # 1. the query length is 1, in which case we will not do iterative updates. Therefore, the kv lacks the current
        #    query length and we need to fill it with zeros.
        # 2. the query length is greater than 1, in which case we will do iterative updates and the kv will have the
        #    correct query length.
        key_states, value_states = past_key_value.layer_get(
            self.layer_type.attends_to,
            zerofill=self.layer_type.attends_top and q_len == 1,
        )

        # handle GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # diagonal mask from the right bottom corner
        if self.config.force_nodiag or self.layer_type.attends_top:
            kv_len = key_states.size(2)
            mask = attn_weights.new_full((q_len, kv_len), torch.finfo(attn_weights.dtype).min)
            mask = mask.tril(diagonal=kv_len - q_len).triu(diagonal=kv_len - q_len)
            attn_weights = attn_weights + mask

        # sliding window mask
        if self.sliding_window:
            kv_len = key_states.size(2)
            mask = attn_weights.new_full((q_len, kv_len), torch.finfo(attn_weights.dtype).min)
            mask = mask.tril(diagonal=kv_len - q_len - self.sliding_window)
            attn_weights = attn_weights + mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LCKVLlamaFlashAttention2(LCKVLlamaAttention):
    """
    LCKV Attention may not need to initialize weights for the key and value projections.
    """

    def __init__(self, config: LCKVLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[LayerCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = apply_rotary(query_states, cos, sin)

        # compute key and value states
        if self.layer_type.computes_kv:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_states = apply_rotary(key_states, cos, sin)

            if isinstance(past_key_value, Cache):
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            past_key_value.layer_set(self.layer_idx, key_states, value_states)

        # get the cached key and value states
        # if the layer attends to the top layers, there are two cases:
        # 1. the query length is 1, in which case we will not do iterative updates. Therefore, the kv lacks the current
        #    query length and we need to fill it with zeros.
        # 2. the query length is greater than 1, in which case we will do iterative updates and the kv will have the
        #    correct query length.
        key_states, value_states = past_key_value.layer_get(
            self.layer_type.attends_to,
            zerofill=self.layer_type.attends_top and q_len == 1,
        )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
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

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=self.sliding_window,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
            no_diag=(self.config.force_nodiag or self.layer_type.attends_top),
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


LCKV_LLAMA_ATTENTION_CLASSES = {
    "eager": LCKVLlamaAttention,
    "flash_attention_2": LCKVLlamaFlashAttention2,
}


class LCKVLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LCKVLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LCKV_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)


class LCKVLlamaPreTrainedModel(LlamaPreTrainedModel):
    config_class = LCKVLlamaConfig
    supports_gradient_checkpointing = False # not tested yet
    _no_split_modules = ["LCKVLlamaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = False


class LCKVLlamaModel(LCKVLlamaPreTrainedModel, LlamaModel):
    def __init__(self, config: LCKVLlamaConfig):
        LCKVLlamaPreTrainedModel.__init__(self, config)
        LlamaModel.__init__(self, copy.deepcopy(config)) # copy config to avoid modifying the original
        self.layers = nn.ModuleList([LCKVLlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.parser = LayerTypeParser(config.layer_types)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LayerCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # build the cache object
        if not isinstance(past_key_values, LayerCache):
            placeholder = inputs_embeds.new_zeros(
                inputs_embeds.shape[0],
                self.config.num_key_value_heads,
                1,
                getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
            )

            if past_key_values is None:
                past_key_values = LayerCache()
            elif isinstance(past_key_values, Cache):
                past_key_values = AutoLayerCache.from_cache(past_key_values)
            else:
                raise NotImplementedError("Only DynamicCache is supported for now.")

            past_key_values.setup(placeholder)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if isinstance(past_key_values, Cache) else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # whether to forward sequentially
        use_sequential = (
            self.config.use_sequential
            or inputs_embeds.shape[1] <= self.config.forward_passes + self.config.backward_passes
            and self.parser.attends_top()
        )

        if use_sequential:

            iteration_outputs = self._modeling_sequential(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_hidden_states=output_hidden_states,
            )

        else:

            # initialize the cache
            past_key_values.initialize(self.parser, inputs_embeds.shape[1])

            # we need to do forward passes based on a plan if the input is a prompt
            plan = self.parser.iteration_plan(self.config.forward_passes, self.config.backward_passes)

            iteration_outputs = self._modeling_with_plan(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_hidden_states=output_hidden_states,
                modeling_plan=plan,
            )

        hidden_states = iteration_outputs.last_hidden_state
        all_hidden_states = iteration_outputs.hidden_states
        all_self_attns = iteration_outputs.attentions
        next_decoder_cache = iteration_outputs.past_key_values

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

    def _iterate_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LayerCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = False,
        layer_slice: Optional[slice] = None,
    ) -> BaseModelOutputWithPast:
        """
        Iterates over the layers of the model, calling each layer in turn.
        """
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # layers to compute
        if layer_slice is None:
            layer_slice = slice(None)

        for decoder_layer in self.layers[layer_slice]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        next_cache = next_decoder_cache if use_cache else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _modeling_with_plan(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LayerCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = False,
        modeling_plan: List[IterStep] = None,
    ) -> BaseModelOutputWithPast:
        """
        Given a plan, iteratively update the hidden states.
        """
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for step in modeling_plan:
            end = len(self.layers) if step.layer_slice.stop is None else step.layer_slice.stop
            iteration_func = self._iterate_layers if step.requires_grad else torch.no_grad()(self._iterate_layers)

            if isinstance(past_key_values, Cache):
                past_key_values._update = step.update

            iteration_outputs = iteration_func(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_hidden_states=output_hidden_states,
                layer_slice=step.layer_slice
            )

            # Update the hidden states cache
            if step.update:
                hidden_states = iteration_outputs.last_hidden_state

            if output_hidden_states:
                all_hidden_states = all_hidden_states[:end] + iteration_outputs.hidden_states

            if output_attentions:
                all_self_attns = all_self_attns[:end] + iteration_outputs.attentions

            if use_cache:
                next_decoder_cache = iteration_outputs.past_key_values

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _modeling_sequential(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LayerCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> BaseModelOutputWithPast:
        """
        Sequentially update the hidden states, token by token.
        """
        seq_len = hidden_states.shape[1]
        last_hidden_state = []
        all_hidden_states = []
        all_self_attns = []

        for i in range(seq_len):
            m_hidden_states = hidden_states[:, i:i+1]
            m_attention_mask = (
                (attention_mask[:, : i + 1] if attention_mask.ndim == 2 else attention_mask[:, :, i : i + 1])
                if attention_mask is not None
                else None
            )
            m_position_ids = position_ids[:, i:i+1] if position_ids is not None else None
            m_cache_position = cache_position[i:i+1] if cache_position is not None else None
            m_position_embeddings = (
                position_embeddings[0][:, i:i+1],
                position_embeddings[1][:, i:i+1]
            )

            outputs = self._iterate_layers(
                m_hidden_states,
                attention_mask=m_attention_mask,
                position_ids=m_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=m_cache_position,
                position_embeddings=m_position_embeddings,
                output_hidden_states=output_hidden_states
            )

            last_hidden_state.append(outputs.last_hidden_state)

            if output_hidden_states:
                all_hidden_states.append(outputs.hidden_states)

            if output_attentions:
                all_self_attns.append(outputs.attentions)

            if use_cache:
                past_key_values = outputs.past_key_values

        last_hidden_state = torch.cat(last_hidden_state, dim=1)

        if output_hidden_states:
            all_hidden_states = [
                torch.cat([hs[i] for hs in all_hidden_states], dim=1) for i in range(len(all_hidden_states[0]))
            ]

        if output_attentions:
            # TODO: deal with attention outputs for non-flash-attention implmentations
            all_self_attns = all_self_attns[-1]

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """fix this function to handle layer cache"""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if isinstance(past_key_values, Cache) else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LCKVLlamaForCausalLM(LCKVLlamaPreTrainedModel, LlamaForCausalLM):
    def __init__(self, config):
        LCKVLlamaPreTrainedModel.__init__(self, config)
        LlamaForCausalLM.__init__(self, copy.deepcopy(config)) # copy config to avoid modifying the original
        self.model = LCKVLlamaModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        """fix this function to handle sink cache"""
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if isinstance(past_key_values, Cache):
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if isinstance(past_key_values, Cache):

                if getattr(past_key_values, "build_position_ids_based_on_cache", False):
                    cur_cache_length = past_key_values.get_seq_length()
                    position_ids = position_ids[:, cur_cache_length :cur_cache_length + input_ids.shape[1]]
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
