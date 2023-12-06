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
""" LLaMA model configuration"""
from transformers.models.llama.configuration_llama import LlamaConfig as _LlamaConfig


class BestLlamaConfig(_LlamaConfig):

    model_type = "best-llama"

    def __init__(
        self,
        kv_pattern: str = "use_kv",
        train_encoder: bool = True,
        **kwargs,
    ):
        """
        Args:
            use_sweet (`bool`, *optional*, defaults to False):
                Use mutli-model strategy introduced in *Finding the SWEET Spot: Analysis
                and Improvement of Adaptive Inference in Low Resource Settings* 
                (https://aclanthology.org/2023.acl-long.829). The gredient in later blocks
                will not affect former blocks.
            kv_pattern (`str`, *optional*, defaults to "use_kv"):
                The pattern to use key-value during inference. The value should be one of 
                "use_kv", "proj_kv", "use_hidden".
                - "use_kv": use the key-value pair in the last layer as the kv cache in all
                    transformer blocks.
                - "proj_kv": use the key-value pair in the last layer as the kv cache in all
                    transformer blocks, but will pass a trainable projection to simulate the
                    original kv.
                - "use_hidden": use the hidden vector in the last layer to recompute the
                    key-value pair in each transformer blocks.
        """
        super().__init__(**kwargs)
        self.kv_pattern = kv_pattern
        self.train_encoder = train_encoder

LlamaConfig = BestLlamaConfig