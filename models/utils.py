from typing import List, Optional
from dataclasses import dataclass

import torch

from transformers.modeling_flash_attention_utils import _flash_attention_forward

@dataclass
class IterStep:
    """A helper class for the iteration plan"""
    layer_slice: slice = slice(None)
    requires_grad: bool = True
    update: bool = True

class LayerType:
    def __init__(self, layer_type: str, layer_idx: int = None):
        self._layer_type = layer_type
        self.layer_idx = layer_idx

        # parse the layer type
        self.layer_types = [int(x) for x in self._layer_type.split("_")]
    
    def __len__(self):
        return len(self.layer_types)
    
    def attends_to(self, layer_idx: int = None) -> int:
        """return the layer that the current layer attends to"""
        if layer_idx is None:
            layer_idx = self.layer_idx
        return self.layer_types[layer_idx]
    
    def attends_top(self, layer_idx: int = None) -> bool:
        """whether the layer attends to layers above it"""
        if layer_idx is None:
            layer_idx = self.layer_idx
        return self.layer_types[layer_idx] > layer_idx
    
    def computes_kv(self, layer_idx: int = None) -> bool:
        """whether the layer computes key-value pairs"""
        if layer_idx is None:
            layer_idx = self.layer_idx
        return layer_idx in self.layer_types
    
    def iteration_plan(self, forward_passes: int = 7, backward_passes: int = 2) -> List[IterStep]:
        """
        Return a iteration plan for the layer types. The plan is a list of IterStep objects.
        """
        attends_top = [self.attends_top(i) for i in range(len(self))]

        # if there is no cyclic dependency, return the default plan
        if True not in attends_top:
            return [IterStep()]
        
        # otherwise, return the plan for the cyclic dependency
        low = attends_top.index(True)
        high = max([i for idx, i in enumerate(self.layer_types) if i > idx])
        plan = [
            # warmup step
            IterStep(slice(low)),

            # do several forward passes only to update KVs
            *forward_passes * [IterStep(slice(low, high + 1), requires_grad=False, update=False)],

            # do backward passes to compute gradients
            *(backward_passes - 1) * [IterStep(slice(low, high + 1), update=False)],
            IterStep(slice(low, high + 1)),

            # finish up the rest of the layers
            IterStep(slice(high + 1, None)),
        ]
        return plan
    
    def check(self, num_hidden_layers: int):
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError("The number of layer types should be equal to the number of hidden layers.")
        for i in range(num_hidden_layers):
            if self.layer_types[i] not in range(num_hidden_layers):
                raise ValueError("The layer type should be in the range of the number of hidden layers.")
            # TODO: manually solve the dependency
            if self.layer_types[i] != self.layer_types[self.layer_types[i]]:
                raise ValueError("The layer should only attends to the layers that attends to itself.")


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    no_diag: bool = False,
):
    """
    This function is a wrapper around the _flash_attention_forward function in the
    transformers library. It adds support to mask the diagonal elements of the attention
    matrix. The diagonal mask is used to resolve the cyclic dependencies in the LCKV model.
    """
    prune_query = False
    if no_diag:
        if key_states.size(1) == 1:
            return torch.zeros_like(query_states)

        if key_states.size(1) == query_states.size(1):
            prune_query = True
            query_states = query_states[:, 1:, :, :]

        key_states = key_states[:, :-1, :, :]
        value_states = value_states[:, :-1, :, :]

    result: torch.Tensor = _flash_attention_forward(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        query_length=query_length,
        is_causal=is_causal,
        dropout=dropout,
        position_ids=position_ids,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        deterministic=deterministic,
    )

    if prune_query:
        b, _, h, d = result.size()
        result = torch.cat([result.new_zeros((b, 1, h, d)), result], dim=1)

    return result