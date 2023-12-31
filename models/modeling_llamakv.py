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

from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from .modeling_llama import (
    LlamaModel as _LlamaModel,
    LlamaForCausalLM as _LlamaForCausalLM,
    apply_rotary_pos_emb_q
)
from .configuration_llama import KVLlamaConfig, HiddenLlamaConfig

class LlamaKVForCausalLM(_LlamaForCausalLM):
    config_class = KVLlamaConfig

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
        r"""
        Only calculate the KV loss.
        """
        assert self.config.kv_pattern == "proj_kv"
        assert past_key_values is None
        assert not use_cache

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # calculate the KV loss
        past_key_values = outputs.past_key_values
        loss = 0
        if self.config.loss_weights is None:
            loss_weights = [1.0] * (self.config.num_hidden_layers - self.config.num_warmup_layers)
        else:
            loss_weights = [float(x) for x in self.config.loss_weights.split("_")]
        
        for i in range(self.config.num_warmup_layers, self.config.num_hidden_layers-1):
            gold_key_state, gold_value_state = past_key_values[i]
            last_key_state, last_value_state = past_key_values[-1]
            bsz, num_key_value_heads, seq_len, head_dim = last_key_state.shape
            last_key_state = last_key_state.transpose(1, 2).reshape(bsz, seq_len, num_key_value_heads*head_dim)
            last_value_state = last_value_state.transpose(1, 2).reshape(bsz, seq_len, num_key_value_heads*head_dim)
            pred_key_state = self.model.layers[i].self_attn.last_k_proj(last_key_state)
            pred_value_state = self.model.layers[i].self_attn.last_v_proj(last_value_state)
            pred_key_state = pred_key_state.reshape(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
            pred_value_state = pred_value_state.reshape(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
            kv_loss = F.mse_loss(pred_key_state, gold_key_state) + F.mse_loss(pred_value_state, gold_value_state)
            loss += loss_weights[i - self.config.num_warmup_layers] * kv_loss

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


class LlamaHiddenForCausalLM(_LlamaForCausalLM):
    config_class = HiddenLlamaConfig

    def __init__(self, config):
        super(_LlamaForCausalLM, self).__init__(config)
        self.tgt_model = _LlamaModel(config)

        if self.config.target_type != "hidden":
            self.tgt_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.tgt_lm_head = self.lm_head

        self.post_init()

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
        r"""
        Only calculate the KV loss.
        """
        assert past_key_values is None
        assert not use_cache

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            encoder_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        encoder_outputs_key_values = encoder_outputs.past_key_values[-1]

        outputs = self.tgt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits_bf16 = self.tgt_lm_head(hidden_states)
        logits = logits_bf16.float()

        # loss weights
        if self.config.loss_weights in ("none", None):
            loss_weights = [1.0] * 2
        else:
            loss_weights = [float(x) for x in self.config.loss_weights.split("_")]

        # the loss to mimic KV and final hidden
        gold_key_state, gold_value_state = encoder_outputs_key_values
        pred_key_state, pred_value_state = outputs[1][-1]
        loss_kv = F.mse_loss(pred_key_state, gold_key_state) + F.mse_loss(pred_value_state, gold_value_state)
        
        loss_outputs = 0
        if self.config.target_type == "hidden":
            loss_outputs = F.mse_loss(hidden_states, encoder_outputs[0])
        elif self.config.target_type == "logits":
            with torch.no_grad():
                encoder_logits = self.lm_head(encoder_outputs[0])
            loss_outputs = F.mse_loss(logits_bf16, encoder_logits)
        elif self.config.target_type == "outputs":
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_outputs = loss_fct(shift_logits, shift_labels)

        loss = loss_weights[0] * loss_kv + loss_weights[1] * loss_outputs

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

        outputs = self.tgt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            encoder_outputs=True,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # manually set the key value
        if use_cache:
            if self.config.kv_pattern == 'use_hidden':
                memory = outputs[0]
                if past_key_values is not None:
                    memory = torch.cat([past_key_values[-1][0], memory], dim=1)
                memory = (memory, )
            else:
                memory = outputs[1][-1]
            new_past_key_values = outputs[1]
            new_past_key_values = new_past_key_values[:self.config.num_warmup_layers] + \
                (memory, )*(len(new_past_key_values) - self.config.num_warmup_layers)
            if return_dict:
                outputs.past_key_values = new_past_key_values
            else:
                outputs = tuple(outputs[0], new_past_key_values, *outputs[2:])

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.tgt_lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.tgt_lm_head(hidden_states)
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