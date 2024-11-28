import re
from dataclasses import dataclass
from typing import List, Optional

import torch

from transformers.modeling_flash_attention_utils import _flash_attention_forward


@dataclass
class IterStep:
    """A helper class for the iteration plan"""
    layer_slice: slice = slice(None)
    requires_grad: bool = True
    update: bool = True

@dataclass
class LayerType:
    """A helper class to collect the layer type information"""
    layer_idx: int
    use_sliding_window: bool
    attends_to: int
    attends_top: bool
    computes_kv: bool

class LayerTypeParser:
    """
    A helper class to parse the layer type string and provide some useful methods.

    Arguments:
        layer_type (str): A string of integers separated by underscores. The i-th integer
            means the layer will use the key-value pair in the i-th layer as the kv cache.
            Special characters may be placed after the integers:
            - `s` means the layer will use sliding window attention.

    >>> layer_type = LayerTypeParser("0_0_0_5s_5s_5s_8_8_8")[3]
    >>> layer_type.attends_to
    5
    >>> layer_type.attends_top
    True
    >>> layer_type.use_sliding_window
    True
    """
    def __init__(self, layer_type: str):
        self._layer_type = layer_type

        # parse the layer type
        self.layer_indices = []
        self.sliding_window = []
        for s in layer_type.split("_"):
            layer_idx, sliding_window = re.match(r"^(\d+)(s)?$", s).groups()
            self.layer_indices.append(int(layer_idx))
            self.sliding_window.append(bool(sliding_window))

    def __len__(self):
        return len(self.layer_indices)

    def __getitem__(self, layer_idx: int) -> LayerType:
        """return the layer type information for the given layer index"""
        return LayerType(
            layer_idx=layer_idx,
            use_sliding_window=self.sliding_window[layer_idx],
            attends_to=self.layer_indices[layer_idx],
            attends_top=self.layer_indices[layer_idx] > layer_idx,
            computes_kv=layer_idx in self.layer_indices,
        )

    def use_sliding_window(self) -> bool:
        """whether there exists a layer that uses sliding window attention"""
        return any(self.sliding_window)

    def attends_top(self) -> bool:
        """whether there exists a layer that attends to layers above it"""
        return any(self.layer_indices[i] > i for i in range(len(self)))

    def iteration_plan(self, forward_passes: int = 7, backward_passes: int = 2) -> List[IterStep]:
        """
        Return a iteration plan for the layer types. The plan is a list of IterStep objects.
        """
        # if there is no cyclic dependency, return the default plan
        if not self.attends_top():
            return [IterStep()]

        # otherwise, return the plan for the cyclic dependency
        plan = []
        i = 0
        while i < len(self):

            # if the layer attends to top layers, resolve the cyclic dependency
            if self[i].attends_top:

                # find the top layer in the cyclic dependency
                top = self[i].attends_to
                while top < max(self.layer_indices[i: top + 1]):
                    top = max(self.layer_indices[i: top + 1])
                top += 1

                # create iteration plan for this group
                layer_slice = slice(i, top)
                plan.extend([
                    *forward_passes * [IterStep(layer_slice, requires_grad=False, update=False)],
                    *(backward_passes - 1) * [IterStep(layer_slice, update=False)],
                    IterStep(layer_slice)
                ])

            # otherwise, create a default plan
            else:

                top = i + 1
                while top < len(self) and not self[top].attends_top:
                    top += 1
                plan.append(IterStep(slice(i, top)))

            # update the index
            i = top

        return plan

    def check(self, num_hidden_layers: int):
        """Check if the layer type is valid"""
        if len(self.layer_indices) != num_hidden_layers:
            raise ValueError("The number of layer types should be equal to the number of hidden layers.")
        for i in range(num_hidden_layers):
            if self.layer_indices[i] not in range(num_hidden_layers):
                raise ValueError("The layer type should be in the range of the number of hidden layers.")


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
            b, l, _, d = value_states.size()
            _, _, h, _ = query_states.size()
            return value_states.new_zeros((b, l, h, d))

        if key_states.size(1) == query_states.size(1):
            prune_query = True
            query_states = query_states[:, 1:, :, :]

        key_states = key_states[:, :-1, :, :]
        value_states = value_states[:, :-1, :, :]

        if sliding_window is not None:
            sliding_window = sliding_window - 1

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
