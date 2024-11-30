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
""" LCKV LLaMA model configuration"""
from transformers.models.llama.configuration_llama import LlamaConfig

from .utils import LayerTypeParser


class LCKVLlamaConfig(LlamaConfig):

    model_type = "lckv-llama"

    def __init__(
        self,
        layer_types: str = None,
        forward_passes: int = 7,
        backward_passes: int = 2,
        sliding_window: int = 4096,
        use_sequential: bool = False,
        force_nodiag: bool = False,
        **kwargs,
    ):
        """
        Initialize a LCKV LLaMA configuration. Instantiating a configuration with the defaults
        will yield a similar configuration to that of the LLaMA-7B with the standard transformer
        training scheme.

        Args:
            layer_types (`str`, *optional*):
                A string of integers separated by underscores. The i-th integer means the layer
                will use the key-value pair in the i-th layer as the kv cache. Special characters
                may be placed after the integers:
                - `s` means the layer will use sliding window attention.
                The default value is "0_1_2_..." till the number of layers in the current config.
            forward_passes (`int`, *optional*, defaults to 7):
                The number of forward passes during training and prompt encoding. Equivlent
                to `m` in the paper.
            backward_passes (`int`, *optional*, defaults to 2):
                The number of backward passes during training and prompt encoding. Equivlent
                to `b` in the paper.
            sliding_window (`int`, *optional*, defaults to 4096):
                Sliding window attention window size. If not specified, will default to `4096`.
                It will only be effective if the corresponding layer uses sliding window attention.
            use_sequential (`bool`, *optional*, defaults to False):
                Whether to do forwarding sequentially, token by token. Useful for testing purpose
                for models with cyclic dependency. Also can be used for sequential training.
            force_nodiag (`bool`, *optional*, defaults to False):
                Whether to force the model to not use the diagonal attention. By default, the model
                will mask the diagonal attention only in layers necessary. If set to `True`, the model
                will never use the diagonal attention in any layer. This is mainly for backward compatibility.
        """
        super().__init__(**kwargs)
        self.layer_types = layer_types
        self.forward_passes = forward_passes
        self.backward_passes = backward_passes
        self.sliding_window = sliding_window
        self.use_sequential = use_sequential
        self.force_nodiag = force_nodiag

        if self.layer_types is None:
            self.layer_types = "_".join(map(str, range(self.num_hidden_layers)))

        # post check
        LayerTypeParser(self.layer_types).check(self.num_hidden_layers)
