from typing import List, Dict, Tuple, Optional, Any
import torch

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

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
    
    def iteration_plan(self, forward_passes: int = 7, backward_passes: int = 2) -> List[Tuple[slice, bool]]:
        """
        Return a iteration plan for the layer types. The plan is a list of tuples,
        where each tuple contains the range of the layers that need to be computed
        and whether to calculate the gradients.
        """
        attends_top = [self.attends_top(i) for i in range(len(self))]

        # if there is no cyclic dependency, return the default plan
        if True not in attends_top:
            return [(slice(None), True)]
        
        # otherwise, return the plan for the cyclic dependency
        lowest_dependent = attends_top.index(True)
        highest_provider = max([i for idx, i in enumerate(self.layer_types) if i > idx])
        plan_warmup = [(slice(lowest_dependent), True)]
        plan_forward = [(slice(lowest_dependent, highest_provider + 1), False)] * forward_passes
        plan_backward = [(slice(lowest_dependent, highest_provider + 1), True)] * backward_passes
        plan_finish = [(slice(highest_provider + 1, None), True)]
        return plan_warmup + plan_forward + plan_backward + plan_finish
    
    def check(self, num_hidden_layers: int):
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError("The number of layer types should be equal to the number of hidden layers.")
        for i in range(num_hidden_layers):
            if self.layer_types[i] not in range(num_hidden_layers):
                raise ValueError("The layer type should be in the range of the number of hidden layers.")
            # TODO: manually solve the dependency
            if self.layer_types[i] != self.layer_types[self.layer_types[i]]:
                raise ValueError("The layer should only attends to the layers that attends to itself.")


class IndexedCache(Cache):
    cache_class = DynamicCache

    """
    Similar to the `DynamicCache` class, but with the ability to index the cache by layer index. DynamicCache
    assumes that all layers compute KVs, while IndexedCache allows for a more flexible cache structure.
    """
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
    
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx in self.key_cache:
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in sorted(self.key_cache.keys()):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])
    
    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers that compute KVs in the model.
        """
        return len(self.key_cache)
    
    @property
    def min_layer(self) -> int:
        return min(self.key_cache.keys()) if len(self.key_cache) > 0 else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if self.min_layer is not None and layer_idx == self.min_layer:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None:
            layer_idx = self.min_layer

        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            (len(self.key_cache) == 0)  # no cache in any layer
            or (layer_idx not in self.key_cache)  # skipped `layer_idx` and hasn't run a layer with cache after it
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. IndexedCache does not have a maximum length."""
        return None

    @classmethod
    def from_cache(cls, dynamic_cache: DynamicCache, *args, **kwargs) -> "IndexedCache":
        """Converts a dynamic cache into an equivalent `IndexedCache`."""
        cache = cls(*args, **kwargs)

        cache._seen_tokens = dynamic_cache._seen_tokens
        for layer_idx in range(len(dynamic_cache.key_cache)):
            key_states, value_states = dynamic_cache[layer_idx]
            cache.update(key_states, value_states, layer_idx)

        return cache


class LayerCache(torch.nn.Module):
    """
    A cache for storing the key-value pairs for layers.
    """
    def __init__(self) -> None:
        """
        The placeholder is used to expand the key-value pairs if the layer attends to the top layers.
        Size: (batch_size, num_key_value_heads, 1, head_dim)
        """
        super().__init__()
        self.key_layer_cache: Dict[int, torch.Tensor] = {}
        self.value_layer_cache: Dict[int, torch.Tensor] = {}
        self.layer_type = None
        self.placeholder = None

    def setup(self, layer_type: LayerType, placeholder: torch.Tensor):
        """setup the cache, calling this function is necessary if there is a layer that attends to the top layers"""
        self.layer_type = layer_type
        self.placeholder = placeholder

    def initialize(self, sequence_length: int):
        """initialize the cache"""
        if self.layer_type is None:
            # TODO: add a warning
            return

        layers_to_init = set(self.layer_type.attends_to(idx) for idx in range(len(self.layer_type)) if self.layer_type.attends_to(idx) > idx)

        if layers_to_init:
            b, h, _, d = self.placeholder.size()
            init_kvs = self.placeholder.new_zeros((b, h, sequence_length, d))

            for layer_idx in layers_to_init:
                self.layer_append(layer_idx, init_kvs, init_kvs)

    def layer_get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states = self.key_layer_cache.get(layer_idx, None)
        value_states = self.value_layer_cache.get(layer_idx, None)

        if self.layer_type is not None and self.layer_type.attends_top(layer_idx):
            if key_states is None:
                key_states = self.placeholder
                value_states = self.placeholder
            else:
                key_states = torch.cat([self.placeholder, key_states], dim=1)
                value_states = torch.cat([self.placeholder, value_states], dim=1)
        
        return key_states, value_states

    def layer_set(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        self.key_layer_cache[layer_idx] = key
        self.value_layer_cache[layer_idx] = value

    def layer_append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        if layer_idx not in self.key_layer_cache:
            self.key_layer_cache[layer_idx] = key
            self.value_layer_cache[layer_idx] = value
        else:
            self.key_layer_cache[layer_idx] = torch.cat([self.key_layer_cache[layer_idx], key], dim=2)
            self.value_layer_cache[layer_idx] = torch.cat([self.value_layer_cache[layer_idx], value], dim=2)


class LayerIndexedCache(LayerCache, IndexedCache):
    """
    A cache for storing the key-value pairs for layers, in combination with the ability of standard KV cache.
    """
    def __init__(self) -> None:
        LayerCache.__init__(self)
        IndexedCache.__init__(self)


class AutoLayerCache(torch.nn.Module):
    """
    AutoLayerCache is a module that automatically creates a cache from an existing cache.
    """
    CACHE_MAPPING = {
        DynamicCache: LayerIndexedCache
    }

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_cache(cache)` method."
        )
    
    @classmethod
    def from_cache(cls, cache: Cache, *args, **kwargs):
        """
        Create a new cache from an existing cache. The new cache will have the same type as the original cache.
        """
        cache_type = type(cache)
        if cache_type not in cls.CACHE_MAPPING:
            raise ValueError(f"Cache type {cache_type} is not supported by {cls.__name__}.")

        cache_class = cls.CACHE_MAPPING[cache_type]

        if hasattr(cache_class, "from_cache"):
            return cache_class.from_cache(cache, *args, **kwargs)
        else:
            # we init an empty cache and copy the attributes
            new_cache = cache_class(*args, **kwargs)
            new_cache.__dict__.update(cache.__dict__)
            return new_cache


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
            return torch.zeros_like(value_states)

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