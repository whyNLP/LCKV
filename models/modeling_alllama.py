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
import itertools
from typing import List, Optional, Tuple, Union, Callable

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

from transformers.models.llama.modeling_llama import (
    LlamaModel as _LlamaModel,
    LlamaForCausalLM as _LlamaForCausalLM,
    LlamaDecoderLayer,
    _prepare_4d_causal_attention_mask,
    LlamaRMSNorm,
    LLAMA_INPUTS_DOCSTRING,
    logger
)

from .configuration_llama import LlamaConfig, ALLlamaConfig, CycleLlamaConfig
    

class LlamaModelBase(_LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.initialize_modules(config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def initialize_modules(self, config):
        """
        This function is intended for overriding. It should initialize `self.layers` as a ModuleList, which will
        be called in `self.__init__`.
        """
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        exit_callback: Optional[Callable] = None,
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
            past_key_values_length = past_key_values[0][0].shape[2]

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

        for idx, decoder_layer in zip(range(self.config.num_hidden_layers), itertools.cycle(self.layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if exit_callback:
                outputs = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=next_decoder_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )
                hidden_states, outputs = exit_callback(hidden_states, outputs, idx)
                if outputs is not None:
                    return outputs

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


class LlamaModel(LlamaModelBase):
    config_class = LlamaConfig


class ALLlamaModel(LlamaModelBase):
    config_class = ALLlamaConfig
    def initialize_modules(self, config):
        self.layers = nn.ModuleList([LlamaDecoderLayer(config)])


class CycleLlamaModel(LlamaModelBase):
    config_class = CycleLlamaConfig
    def initialize_modules(self, config):
        if config.num_hidden_layers % config.cycles:
            raise ValueError(f"Number of hidden layers ({config.num_hidden_layers}) must be a multiple of number of cycles ({config.cycles}).")
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers // config.cycles)])


class LlamaForCausalLMBase(_LlamaForCausalLM):
    TSFM_CLASS = LlamaModel

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = self.TSFM_CLASS(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Early exit config
        self.loss_layers = [int(x) if int(x) >= 0 else int(x) + config.num_hidden_layers for x in config.loss_layers.split("_")]
        self.loss_weights = [float(x) for x in config.loss_weights.split("_")]
        self.exit_layers = [int(x) if int(x) >= 0 else int(x) + config.num_hidden_layers for x in config.exit_layers.split("_")]

        # Early exit classifiers
        if config.share_head:
            # reuse params, do not register
            if config.use_ln_head:
                self.lm_heads = [
                    nn.Sequential(
                        self.model.norm,
                        self.lm_head
                    )
                    for _ in range(len(self.loss_weights) - 1)
                ]
            else:
                self.lm_heads = [
                    self.lm_head
                    for _ in range(len(self.loss_weights) - 1)
                ]
        else:
            # create new blocks
            if config.use_ln_head:
                self.lm_heads = nn.ModuleList([
                    nn.Sequential(
                        LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                        nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
                    )
                    for _ in range(len(self.loss_weights) - 1)
                ])
            else:
                self.lm_heads = nn.ModuleList([
                    nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
                    for _ in range(len(self.loss_weights) - 1)
                ])
        
        # Loss func
        self.loss_func = CrossEntropyLoss()

        # custom log
        self._custom_log = dict()

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

        # preparation for early exit
        is_early_exit = False
        loss = None
        _custom_log = dict()
        if not self.training and labels is not None:
            if self.config.exit_strategy in ("confidence", "softmax"):
                # prepare to remember logits
                exited_logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size, dtype=self.dtype, device=input_ids.device)
            elif self.config.exit_strategy == "similarity":
                # prepare to remember hidden states
                exited_hidden_states = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.hidden_size, dtype=self.dtype, device=input_ids.device)
            exited_indicator = torch.zeros(input_ids.size(0), input_ids.size(1), dtype=torch.bool, device=input_ids.device)
        if not self.training:
            # prepare to remember hidden states
            previous_hidden_states = None
        if labels is not None:
            loss = 0.0
            shift_labels = labels[..., 1:].contiguous()
        
        def exit_callback(hidden_states: torch.Tensor, outputs: BaseModelOutputWithPast, i: int):
            nonlocal loss, is_early_exit, exited_logits, exited_hidden_states, exited_indicator, previous_hidden_states, _custom_log, past_key_values

            # we leave the last task for future
            if i == self.config.num_hidden_layers - 1:

                # register custom log
                if not self.training and labels is None:
                    _custom_log.update({
                        'early_exit_layers': self._custom_log.get('early_exit_layers', tuple()) + (i, )
                    })
                
                return hidden_states, None

            # during training, we calculate loss from specific layers
            if self.training:
                if i in self.loss_layers:
                    idx = self.loss_layers.index(i)

                    # detach gradient to prevent backprop
                    if self.config.use_sweet:
                        hidden_states = hidden_states.detach()

                    # calculate logits
                    logits = self.lm_heads[idx](hidden_states)

                    # calculate loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    layer_loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss += self.loss_weights[idx] * layer_loss

                    # add to custom log
                    layer_loss = layer_loss.item()
                    _custom_log.update({f'train_loss_layer_{str(i)}': layer_loss})

            # during inference, we deal with early exit to calculate loss (but not really exit early)
            elif labels is not None:
                if i in self.exit_layers:

                    # reuse exited hidden states
                    if previous_hidden_states is not None:
                        hidden_states[exited_indicator] = previous_hidden_states[exited_indicator]
                    
                    # collect the logits that are ready to exit
                    if self.config.exit_strategy == "confidence":
                        lm_head = self.lm_heads[self.loss_layers.index(i)] if i in self.loss_layers else self.lm_head
                        logits: torch.Tensor = lm_head(hidden_states)
                        exit_entries = logits.softmax(-1).max(-1)[0] >= self.config.exit_threshold
                    
                        exit_entries &= ~exited_indicator
                        exited_indicator |= exit_entries
                        logits = logits.to(dtype=self.dtype)
                        exited_logits[exit_entries] = logits[exit_entries]
                    
                    elif self.config.exit_strategy == "softmax":
                        lm_head = self.lm_heads[self.loss_layers.index(i)] if i in self.loss_layers else self.lm_head
                        logits: torch.Tensor = lm_head(hidden_states)
                        maximums, _ = logits.softmax(-1).topk(2, dim=-1)
                        exit_entries = (maximums[..., 0] - maximums[..., 1]) >= self.config.exit_threshold
                    
                        exit_entries &= ~exited_indicator
                        exited_indicator |= exit_entries
                        logits = logits.to(dtype=self.dtype)
                        exited_logits[exit_entries] = logits[exit_entries]
                    
                    elif self.config.exit_strategy == "similarity":
                        exit_entries = torch.cosine_similarity(hidden_states, previous_hidden_states, dim=-1) >= self.config.exit_threshold

                        exit_entries &= ~exited_indicator
                        exited_indicator |= exit_entries
                        exited_hidden_states[exit_entries] = hidden_states[exit_entries]
            
            # if we are doing real generation, we need to really exit early
            else:
                # we first implement a simple version that only supports batch size = 1
                if hidden_states.size(0) != 1:
                    raise NotImplementedError("Early exit with batch size > 1 is not yet implemented for realy generation.")
                
                if i in self.exit_layers:
                    lm_head = self.lm_heads[self.loss_layers.index(i)] if i in self.loss_layers else self.lm_head
                    logits: torch.Tensor = lm_head(hidden_states)

                    # see if the model is confident enough to exit
                    if self.config.exit_strategy == "confidence":
                        if logits[..., -1, :].softmax(-1).max().item() >= self.config.exit_threshold:
                            is_early_exit = True
                    elif self.config.exit_strategy == "softmax":
                        maximums, _ = logits[..., -1, :].softmax(-1).topk(2, dim=-1)
                        if (maximums[..., 0] - maximums[..., 1]).item() >= self.config.exit_threshold:
                            is_early_exit = True
                    elif self.config.exit_strategy == "similarity":
                        if torch.cosine_similarity(hidden_states[..., -1, :], previous_hidden_states[..., -1, :], dim=-1).item() >= self.config.exit_threshold:
                            is_early_exit = True
                        
                    # ready to exit
                    if is_early_exit:

                        # register custom log
                        _custom_log.update({
                            'early_exit_layers': self._custom_log.get('early_exit_layers', tuple()) + (i, )
                        })

                        # one important thing is to prepare the kv cache, just repeat the last kv
                        # shold be shape: layers x (past_key, past_value)
                        presents = outputs.past_key_values
                        if presents is not None:
                            _, seq_len = input_ids.size()
                            last_key, last_value = presents[-1]
                            last_key, last_value = last_key[..., -seq_len:, :], last_value[..., -seq_len:, :]
                            for j in range(i + 1, self.config.num_hidden_layers):
                                if past_key_values is None:
                                    key, value = last_key, last_value
                                else:
                                    past_key, past_value = past_key_values[j]
                                    key = torch.cat((past_key, last_key), dim=-2)
                                    value = torch.cat((past_value, last_value), dim=-2)
                                presents = presents + ((key, value), )

                        if output_hidden_states:
                            all_hidden_states = outputs.hidden_states + (logits,)
                        else:
                            all_hidden_states = outputs.hidden_states

                        if not return_dict:
                            outputs = tuple(
                                v
                                for v in [logits, presents, all_hidden_states, outputs.attentions]
                                if v is not None
                            )
                        else:
                            outputs = BaseModelOutputWithPast(
                                last_hidden_state=logits,
                                past_key_values=presents,
                                hidden_states=all_hidden_states,
                                attentions=outputs.attentions,
                            )
                        
                        return logits, outputs
            
            # we need to remember the hidden states for early exit
            # XXX: do we need to use clone? from the code for all_hidden_states, I think
            #      it might not be necessary.
            previous_hidden_states = hidden_states

            return hidden_states, None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            exit_callback=exit_callback,
        )

        hidden_states = outputs[0]
        # if self.config.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        #     logits = self.lm_head(hidden_states)
        # logits = logits.float()

        # do the last classification if not early exit
        if not is_early_exit:
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = hidden_states

        # deal with the last layer
        if self.training:
            # calculate loss
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # Flatten the tokens
            layer_loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.loss_weights[-1] * layer_loss
            # add to custom log
            layer_loss = layer_loss.item()
            _custom_log.update({f'train_loss_layer_{str(self.config.num_hidden_layers-1)}': layer_loss})
        elif labels is not None:
            # all entries must exit
            exit_entries = ~exited_indicator
            if self.config.exit_strategy in ("confidence", "softmax"):
                lm_logits = lm_logits.to(dtype=self.dtype)
                exited_logits[exit_entries] = lm_logits[exit_entries]
            elif self.config.exit_strategy == "similarity":
                exited_hidden_states[exit_entries] = previous_hidden_states[exit_entries]
                exited_hidden_states = self.model.norm(exited_hidden_states)
                exited_logits = self.lm_head(exited_hidden_states)

            # calculate loss
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = exited_logits[..., :-1, :].contiguous()
            # Flatten the tokens
            loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        # register custom log
        self._custom_log = _custom_log

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForCausalLM(LlamaForCausalLMBase):
    config_class = LlamaConfig
    TSFM_CLASS = LlamaModel

class ALLlamaForCausalLM(LlamaForCausalLMBase):
    config_class = ALLlamaConfig
    TSFM_CLASS = ALLlamaModel

class CycleLlamaForCausalLM(LlamaForCausalLMBase):
    config_class = CycleLlamaConfig
    TSFM_CLASS = CycleLlamaModel