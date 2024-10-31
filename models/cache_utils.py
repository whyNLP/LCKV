from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers.cache_utils import Cache, DynamicCache, SinkCache

from .utils import LayerTypeParser


class IndexedCache(Cache):
    """
    Similar to the `DynamicCache` class, but with the ability to index the cache by layer index. DynamicCache
    assumes that all layers compute KVs, while IndexedCache allows for a more flexible cache structure.
    """
    build_position_ids_based_on_cache = False

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self._update = True # to prevent the cache from updating when inference with iterations

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

    def is_min_layer(self, layer_idx: int) -> bool:
        return self.min_layer is None or self.min_layer == layer_idx

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
        if self.is_min_layer(layer_idx):
            self._seen_tokens += key_states.shape[-2]

        # Retrieve the cache
        if layer_idx not in self.key_cache:
            new_key_states = key_states
            new_value_states = value_states
        else:
            new_key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # Update the cache
        if self._update:
            self.key_cache[layer_idx] = new_key_states
            self.value_cache[layer_idx] = new_value_states

        return new_key_states, new_value_states

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


class IndexedSinkCache(Cache):
    """
    This is a fix to the SinkCache class in the transformers library. It also allows for the cache to be indexed by
    layer index, similar to the `IndexedCache` class.
    """
    build_position_ids_based_on_cache = True

    def __init__(self, window_length: int = None, num_sink_tokens: int = None) -> None:
        super().__init__()
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self._update = True  # to prevent the cache from updating when inference with iterations

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, offset: int, dtype: torch.dtype, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset not in self.cos_sin_rerotation_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + offset :]
            shifted_cos = cos[self.num_sink_tokens : -offset]
            original_sin = sin[self.num_sink_tokens + offset :]
            shifted_sin = sin[self.num_sink_tokens : -offset]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_rerotation_cache[offset] = (
                rerotation_cos.to(dtype).unsqueeze(0),
                rerotation_sin.to(dtype).unsqueeze(0),
            )
        return self.cos_sin_rerotation_cache[offset]

    @property
    def min_layer(self) -> int:
        return min(self.key_cache.keys()) if len(self.key_cache) > 0 else None

    def is_min_layer(self, layer_idx: int) -> bool:
        return self.min_layer is None or self.min_layer == layer_idx

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if layer_idx is None:
            layer_idx = self.min_layer

        if layer_idx not in self.key_cache:
            return 0

        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

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
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if self.is_min_layer(layer_idx):
            self._seen_tokens += key_states.shape[-2]

        # Update the sin/cos cache, which holds sin/cos values for all possible positions
        if using_rope and self.is_min_layer(layer_idx):
            # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
            # after all RoPE models have a llama-like cache utilization.
            if cos.dim() == 2:
                self._cos_cache = cos
                self._sin_cache = sin
            else:
                if self._cos_cache is None:
                    self._cos_cache = cos[0, ...]
                    self._sin_cache = sin[0, ...]
                elif self._cos_cache.shape[0] < self.window_length + key_states.shape[-2]:
                    self._cos_cache = torch.cat([self._cos_cache[: self.window_length], cos[0, ...]], dim=0)
                    self._sin_cache = torch.cat([self._sin_cache[: self.window_length], sin[0, ...]], dim=0)

        # [bsz, num_heads, seq_len, head_dim]
        if layer_idx not in self.key_cache:
            # Empty cache
            new_key_states = key_states
            new_value_states = value_states

        else:
            # Growing cache
            new_key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if self._update:
            self.key_cache[layer_idx] = new_key_states
            self.value_cache[layer_idx] = new_value_states

        # If the cache is full, we need to shift the cache
        if (seq_length := self.get_seq_length(layer_idx)) > self.window_length:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][:, :, -self.window_length + self.num_sink_tokens :]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    seq_length - self.window_length,
                    key_states.dtype,
                    self._cos_cache[:seq_length],
                    self._sin_cache[:seq_length],
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][:, :, -self.window_length + self.num_sink_tokens :]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep], dim=-2)

        return new_key_states, new_value_states

    @classmethod
    def from_cache(cls, sink_cache: SinkCache, *args, **kwargs) -> "IndexedSinkCache":
        """Converts a dynamic cache into an equivalent `IndexedCache`."""
        cache = cls(*args, **kwargs)

        cache.window_length = sink_cache.window_length
        cache.num_sink_tokens = sink_cache.num_sink_tokens
        cache._seen_tokens = sink_cache._seen_tokens
        cache._cos_cache = sink_cache._cos_cache
        cache._sin_cache = sink_cache._sin_cache
        cache.cos_sin_rerotation_cache = sink_cache.cos_sin_rerotation_cache
        for layer_idx in range(len(sink_cache.key_cache)):
            cache.key_cache[layer_idx] = sink_cache.key_cache[layer_idx]
            cache.value_cache[layer_idx] = sink_cache.value_cache[layer_idx]

        return cache


class IndexedSlidingWindowCache(IndexedCache):
    """
    Similar to the `SlidingWindowCache` class, but with the ability to index the cache by layer index. It is no longer
    a subclass of `StaticCache` as it is dynamic.
    """
    build_position_ids_based_on_cache = False

    def __init__(self, sliding_window: int = None) -> None:
        super().__init__()
        self.sliding_window = sliding_window

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        # Update the number of seen tokens
        if self.is_min_layer(layer_idx):
            self._seen_tokens += key_states.shape[-2]

        # [bsz, num_heads, seq_len, head_dim]
        if layer_idx not in self.key_cache:
            # Empty cache
            new_key_states = key_states
            new_value_states = value_states

        else:
            # Growing cache
            new_key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if self._update:
            self.key_cache[layer_idx] = new_key_states
            self.value_cache[layer_idx] = new_value_states

        # If the cache is full, we need to shift the cache
        if self.get_seq_length(layer_idx) > self.sliding_window:
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -self.sliding_window :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -self.sliding_window :]

        return new_key_states, new_value_states

    def get_max_length(self) -> Optional[int]:
        return self.sliding_window

    @classmethod
    def from_cache(cls, sliding_window_cache: "IndexedSlidingWindowCache", *args, **kwargs) -> "IndexedSlidingWindowCache":
        """This is to override the `from_cache` method in the `IndexedCache` class."""
        cache = cls(*args, **kwargs)

        cache._seen_tokens = sliding_window_cache._seen_tokens
        cache.sliding_window = sliding_window_cache.sliding_window
        for layer_idx in range(len(sliding_window_cache.key_cache)):
            cache.key_cache[layer_idx] = sliding_window_cache.key_cache[layer_idx]
            cache.value_cache[layer_idx] = sliding_window_cache.value_cache[layer_idx]

        return cache


class IndexedHybridCache(IndexedSlidingWindowCache, IndexedCache):
    """
    Hybrid Cache class to be used for models that alternate between a local sliding window attention and global
    attention in every other layer. Under the hood, Hybrid Cache leverages ["IndexedSlidingWindowCache"] for
    sliding window attention and ["IndexedCache"] for global attention.
    """
    build_position_ids_based_on_cache = False

    def __init__(self, parser: LayerTypeParser = None, sliding_window: int = None) -> None:
        super().__init__(sliding_window=sliding_window)
        self.parser = parser

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        if self.parser[layer_idx].use_sliding_window:
            return IndexedSlidingWindowCache.update(self, key_states, value_states, layer_idx, cache_kwargs)
        else:
            return IndexedCache.update(self, key_states, value_states, layer_idx, cache_kwargs)

    def get_max_length(self) -> Optional[int]:
        return IndexedCache.get_max_length(self)

    @classmethod
    def from_cache(cls, hybrid_cache: "IndexedHybridCache", *args, **kwargs) -> "IndexedHybridCache":
        """This is to override the `from_cache` method in the `IndexedSlidingWindowCache` class."""
        cache = cls(*args, **kwargs)

        cache._seen_tokens = hybrid_cache._seen_tokens
        cache.sliding_window = hybrid_cache.sliding_window
        cache.parser = hybrid_cache.parser
        for layer_idx in range(len(hybrid_cache.key_cache)):
            cache.key_cache[layer_idx] = hybrid_cache.key_cache[layer_idx]
            cache.value_cache[layer_idx] = hybrid_cache.value_cache[layer_idx]

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

    def setup(self, placeholder: torch.Tensor):
        """setup the cache, calling this function is necessary if there is a layer that attends to the top layers"""
        self.placeholder = placeholder

    def initialize(self, parser: LayerTypeParser, sequence_length: int):
        """initialize the cache"""
        layers_to_init = {parser[idx].attends_to for idx in range(len(parser)) if parser[idx].attends_top}

        if layers_to_init:
            b, h, _, d = self.placeholder.size()
            init_kvs = self.placeholder.new_zeros((b, h, sequence_length, d))

            for layer_idx in layers_to_init:
                self.layer_append(layer_idx, init_kvs, init_kvs)

    def layer_get(self, layer_idx: int, zerofill: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states = self.key_layer_cache.get(layer_idx, None)
        value_states = self.value_layer_cache.get(layer_idx, None)

        if zerofill:
            if key_states is None:
                key_states = self.placeholder
                value_states = self.placeholder
            else:
                key_states = torch.cat([self.placeholder, key_states], dim=2)
                value_states = torch.cat([self.placeholder, value_states], dim=2)

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


class LayerIndexedSinkCache(LayerCache, IndexedSinkCache):
    """
    A cache for storing the key-value pairs for layers, in combination with the ability of sink KV cache.
    """
    def __init__(self) -> None:
        LayerCache.__init__(self)
        IndexedSinkCache.__init__(self)


class LayerIndexedSlidingWindowCache(LayerCache, IndexedSlidingWindowCache):
    """
    A cache for storing the key-value pairs for layers, in combination with the ability of sliding window KV cache.
    """
    def __init__(self) -> None:
        LayerCache.__init__(self)
        IndexedSlidingWindowCache.__init__(self)


class LayerIndexedHybridCache(LayerCache, IndexedHybridCache):
    """
    A cache for storing the key-value pairs for layers, in combination with the ability of hybrid KV cache.
    """
    def __init__(self) -> None:
        LayerCache.__init__(self)
        IndexedHybridCache.__init__(self)


class AutoLayerCache(torch.nn.Module):
    """
    AutoLayerCache is a module that automatically creates a cache from an existing cache.
    """
    CACHE_MAPPING = {
        DynamicCache: LayerIndexedCache,
        SinkCache: LayerIndexedSinkCache,
        IndexedSlidingWindowCache: LayerIndexedSlidingWindowCache,
        IndexedHybridCache: LayerIndexedHybridCache,
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
