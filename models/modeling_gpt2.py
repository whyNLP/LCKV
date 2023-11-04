# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

import itertools
from typing import Optional, Tuple, Union, Callable

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
    GPT2_INPUTS_DOCSTRING,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as _GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as _GPT2LMHeadModel
from .configuration_gpt2 import GPT2Config, ALGPT2Config, CycleGPT2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"


class GPT2ModelBase(_GPT2Model):
    """
    This implements a variant on GPT2Model, including:
     - Allow parameter sharing stretegy, such as ALGPT2 or CycleGPT2;
     - Leave room for Early Exit callback.

    This might break model parallel.
    """
    def __init__(self, config):
        super(_GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.initialize_modules(config)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def initialize_modules(self, config):
        """
        This function is intended for overriding. It should initialize `self.h` as a ModuleList, which will
        be called in `self.__init__`.
        """
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
    
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        exit_callback: Optional[Callable] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(itertools.cycle(self.h), past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
            
            if exit_callback:
                outputs = BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=presents,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attentions,
                    cross_attentions=all_cross_attentions,
                )
                hidden_states, outputs = exit_callback(hidden_states, outputs, i)
                if outputs is not None:
                    return outputs

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2Model(GPT2ModelBase):
    config_class = GPT2Config


class ALGPT2Model(GPT2ModelBase):
    config_class = ALGPT2Config
    def initialize_modules(self, config):
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=0)])


class CycleGPT2Model(GPT2ModelBase):
    config_class = CycleGPT2Config
    def initialize_modules(self, config):
        if config.num_hidden_layers % config.cycles:
            raise ValueError(f"Number of hidden layers ({config.num_hidden_layers}) must be a multiple of number of cycles ({config.cycles}).")
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers // config.cycles)])


class GPT2LMHeadModelBase(_GPT2LMHeadModel):
    TSFM_CLASS = GPT2Model

    def __init__(self, config):
        super(_GPT2LMHeadModel, self).__init__(config)
        self.transformer = self.TSFM_CLASS(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
                        self.transformer.ln_f,
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
                        nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
                        nn.Linear(config.n_embd, config.vocab_size, bias=False) 
                    )
                    for _ in range(len(self.loss_weights) - 1)
                ])
            else:
                self.lm_heads = nn.ModuleList([
                    nn.Linear(config.n_embd, config.vocab_size, bias=False) 
                    for _ in range(len(self.loss_weights) - 1)
                ])
        
        # Loss func
        self.loss_func = CrossEntropyLoss()

        # custom log
        self._custom_log = dict()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
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
                exited_hidden_states = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.n_embd, dtype=self.dtype, device=input_ids.device)
            exited_indicator = torch.zeros(input_ids.size(0), input_ids.size(1), dtype=torch.bool, device=input_ids.device)
        if not self.training:
            # prepare to remember hidden states
            previous_hidden_states = None
        if labels is not None:
            loss = 0.0
            shift_labels = labels[..., 1:].contiguous()
        
        def exit_callback(hidden_states: torch.Tensor, outputs: BaseModelOutputWithPastAndCrossAttentions, i: int):
            nonlocal loss, is_early_exit, exited_logits, exited_hidden_states, exited_indicator, previous_hidden_states, _custom_log

            # we leave the last task for future
            if i == self.config.num_hidden_layers - 1:
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

                        # one important thing is to prepare the kv cache, just repeat the last kv
                        past_key_values = outputs.past_key_values
                        if past_key_values is not None:
                            last_kv = past_key_values[-1]
                            past_key_values = past_key_values + tuple(last_kv for _ in range(self.config.num_hidden_layers - i - 1))

                        if output_hidden_states:
                            all_hidden_states = outputs.hidden_states + (logits,)
                        else:
                            all_hidden_states = outputs.hidden_states

                        if not return_dict:
                            outputs = tuple(
                                v
                                for v in [logits, past_key_values, all_hidden_states, outputs.attentions, outputs.cross_attentions]
                                if v is not None
                            )
                        else:
                            outputs = BaseModelOutputWithPastAndCrossAttentions(
                                last_hidden_state=logits,
                                past_key_values=past_key_values,
                                hidden_states=all_hidden_states,
                                attentions=outputs.attentions,
                                cross_attentions=outputs.cross_attentions,
                            )
                        
                        return logits, outputs
            
            # we need to remember the hidden states for early exit
            # XXX: do we need to use clone? from the code for all_hidden_states, I think
            #      it might not be necessary.
            previous_hidden_states = hidden_states

            return hidden_states, None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            exit_callback=exit_callback,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

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
                exited_hidden_states = self.transformer.ln_f(exited_hidden_states)
                exited_logits = self.lm_head(exited_hidden_states)

            # calculate loss
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = exited_logits[..., :-1, :].contiguous()
            # Flatten the tokens
            loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        # register custom log
        self._custom_log = _custom_log

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT2LMHeadModel(GPT2LMHeadModelBase):
    TSFM_CLASS = GPT2Model

class ALGPT2LMHeadModel(GPT2LMHeadModelBase):
    TSFM_CLASS = ALGPT2Model

class CycleGPT2LMHeadModel(GPT2LMHeadModelBase):
    TSFM_CLASS = CycleGPT2Model