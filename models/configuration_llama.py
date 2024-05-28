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

class OptLlamaConfig(_LlamaConfig):

    model_type = "opt-llama"

    def __init__(
        self,
        num_trained_encoders: int = 1,
        train_kv: bool = False,
        num_encoders: int = 8,
        layer_types: str = None,
        target_layer: int = -1,
        **kwargs,
    ):
        """
        Args:
            train_last_encoder (`str`, *optional*, defaults to "none"):
                Deprecated. Use `num_trained_encoder` instead.
                Whether to train the last encoder. The value should be one of "none",
                "encoder" or an integer for the number of encoders to train.
            num_trained_encoders (`int`, *optional*, defaults to 1):
                Number of encoders to train. The last `num_trained_encoders` will be
                trained. Equivlent to `b-1` in the paper.
            train_kv (`bool`, *optional*, defaults to False):
                Whether to train the key-value pair. If set to True, the loss will be
                added with the MSE loss of the key-value pair in the last layer before
                and after the decoder. This helps the KV cache to converge so that the
                training and inference will be consistent, but hurts the performance.
            num_encoders (`int`, *optional*, defaults to 8):
                The number of encoders. x encoders will ensure the starting x tokens in
                prediction is consistent with training. Equivlent to `m+b-1` in the paper.
            num_warmup_layers (`int`, *optional*, defaults to 0):
                Deprecated. Use `layer_types` and `target_layer` instead.
                The number of transformer blocks that will use the key-value pair in the
                original layers as the kv cache. The rest of the transformer blocks will
                use the key-value pair in the last layer as the kv cache.
            layer_types (`str`, *optional*, defaults to ""):
                The type of each layer. The value should be a underscore separated string
                of integers. The value "0" means the layer will use the key-value pair in
                the original layers as the kv cache. The value "1" means the layer will
                use the key-value pair in (possibly) the last layer as the kv cache. 
                The value "2" means the layer will use the key-value pair during training,
                but also generate the key-value pair for other layers to use. The default
                value is all "0".
            target_layer (`int`, *optional*, defaults to -1):
                The layer to generate key-value pair. The value should be in 
                [0, num_hidden_layers). The default value is -1, which means the last layer.
        """
        # deal with deprecated args
        if "train_last_encoder" in kwargs:
            train_last_encoder = kwargs.pop("train_last_encoder")
            if train_last_encoder == "none":
                num_trained_encoders = 0
            elif train_last_encoder == "encoder":
                num_trained_encoders = 1
            else:
                num_trained_encoders = int(train_last_encoder)
        
        if "num_warmup_layers" in kwargs:
            num_warmup_layers = kwargs.pop("num_warmup_layers")
            num_hidden_layers = kwargs.get("num_hidden_layers")
            layer_types = ["0"] * num_warmup_layers + ["1"] * (num_hidden_layers - num_warmup_layers - 1) + ["2"]
            layer_types = "_".join(layer_types)
            target_layer = num_hidden_layers - 1
        
        super().__init__(**kwargs)
        self.num_trained_encoders = num_trained_encoders
        self.train_kv = train_kv
        self.num_encoders = num_encoders
        self.layer_types = layer_types
        self.target_layer = target_layer

        if self.layer_types is None:
            self.layer_types = "_".join(["0"]*self.num_hidden_layers)

        # post check
        num_hidden_layers = self.num_hidden_layers
        layer_types = [int(x) for x in self.layer_types.split("_")]
        if len(layer_types) != num_hidden_layers:
            raise ValueError("The number of layer types should be equal to the number of hidden layers.")
        for i in range(num_hidden_layers):
            if layer_types[i] not in (0, 1, 2):
                raise ValueError("The layer type should be one of 0, 1 and 2.")
            if layer_types[i] == 2 and target_layer % num_hidden_layers != i:
                raise ValueError("The target layer should be the layer of type 2.")
        if layer_types.count(2) > 1:
            raise ValueError("Only one layer can be type 2.")

class ClaLlamaConfig(_LlamaConfig):
    model_type = "cla-llama"

    def __init__(
        self,
        layer_types: str = None,
        **kwargs,
    ):
        """
        This is an implementation of the Cross-Layer Attention (CLA) model.

        Args:
            layer_types (`str`, *optional*, defaults to ""):
                The type of each layer. The value should be a underscore separated string
                of integers. The value "0" means the layer will use the key-value pair in
                the original layers as the kv cache. The value "1" means the layer will
                use the key-value pair in the nearest lower layer as the kv cache. 
                The value "2" is the same as "0", but to be consistent with LCKV we name
                the bottom layer of each group as "2". The default value is all "0".
        
                Example:
                - "2_1_2_1_2_1_2_1_2_1" is a CLA-2 model with 10 layers.
                - "0_2_1_1_2_1_1_2_1_1" is a CLA-3 model with 10 layers.
        
        See more info in Figure 2 of the paper "Reducing Transformer Key-Value Cache
        Size with Cross-Layer Attention", http://arxiv.org/abs/2405.12981
        """
        super().__init__(**kwargs)
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = "_".join(["0"]*self.num_hidden_layers)

        # post check
        num_hidden_layers = self.num_hidden_layers
        layer_types = [int(x) for x in self.layer_types.split("_")]
        if len(layer_types) != num_hidden_layers:
            raise ValueError("The number of layer types should be equal to the number of hidden layers.")
        for i in range(num_hidden_layers):
            if layer_types[i] not in (0, 1, 2):
                raise ValueError("The layer type should be one of 0, 1 and 2.")
        if layer_types[0] == 1:
            raise ValueError("The first layer should be type 0 or 2. It must calculates the KV.")
