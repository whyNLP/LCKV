import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import warnings
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .modeling_algpt2 import ALGPT2Model, ALGPT2LMHeadModel, ALGPT2Config
from .modeling_algpt2 import (
    BaseModelOutputWithPastAndCrossAttentions,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    GPT2_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC
)

# ALGPT
from .modeling_algpt2 import ALGPT2LMHeadModel, ALGPT2Config

class ALGPT2EarlyExitConfig(ALGPT2Config):
    model_type = "algpt2-early-exit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_exit = False

class ALGPT2EarlyExitModel(ALGPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ALGPT2Model(config)
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for _ in range(config.n_layer)
        ])

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
        Current implementation only adds up the loss from all the layers.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not output_hidden_states:
            warnings.warn(
                "This implementation requires to output hidden states as it needs to calculate the loss for each layer. ",
                UserWarning
            )
            output_hidden_states = True

        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
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
        )
        hidden_states = transformer_outputs.hidden_states[1:] # exclude input embeddings

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            for hidden_state in hidden_states:
                hidden_state = hidden_state.to(self.lm_head.weight.device)

        # Calculate logits for each layer
        lm_logits = tuple(lm_head(hidden_state) for hidden_state, lm_head in zip(hidden_states, self.lm_heads))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits[0].device)
            # Shift so that tokens < n predict n
            shift_labels = labels[..., 1:].contiguous()

            loss = 0
            for lm_logit in lm_logits:
                # Shift so that tokens < n predict n
                shift_logits = lm_logit[..., :-1, :].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        lm_logits = lm_logits[-1]

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