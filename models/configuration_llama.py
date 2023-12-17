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
        mask_diagonal: bool = False,
        num_encoders: int = 1,
        num_warmup_layers: int = 0,
        **kwargs,
    ):
        """
        Args:
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
            train_encoder (`bool`, *optional*, defaults to True):
                Whether to train the encoder. If set to False, the encoder will be detached
                from the computation graph when calculating the gradients.
            mask_diagonal (`bool`, *optional*, defaults to False):
                Whether to mask the diagonal of the attention matrix. This is useful when
                the model is used for sequence generation. The training and inference
                should be consistent.
            num_encoders (`int`, *optional*, defaults to 1):
                The number of encoders. x encoders will ensure the starting x tokens in
                prediction is consistent with training.
            num_warmup_layers (`int`, *optional*, defaults to 0):
                The number of transformer blocks that will use the key-value pair in the
                original layers as the kv cache. The rest of the transformer blocks will
                use the key-value pair in the last layer as the kv cache.
        """
        super().__init__(**kwargs)
        self.kv_pattern = kv_pattern
        self.train_encoder = train_encoder
        self.mask_diagonal = mask_diagonal
        self.num_encoders = num_encoders
        self.num_warmup_layers = num_warmup_layers


class LlamaConfigBase(_LlamaConfig):
    def __init__(
        self,
        use_sweet: bool = False,
        use_ln_head: bool = True,
        share_head: bool = False,
        exit_strategy: str = "confidence",
        loss_layers: str = "-1",
        loss_weights: str = "1.0",
        exit_layers: str = "-1",
        exit_threshold: float = 1.0,
        **kwargs
    ):
        """
        Args:
            use_sweet (`bool`, *optional*, defaults to False):
                Use mutli-model strategy introduced in *Finding the SWEET Spot: Analysis
                and Improvement of Adaptive Inference in Low Resource Settings* 
                (https://aclanthology.org/2023.acl-long.829). The gredient in later blocks
                will not affect former blocks.
            use_ln_head (`bool`, *optional*, defaults to True):
                Add a layer norm before each prediction head. This is consistent with GPT2
                structure, which has a layer norm at the end of transformer blocks.
            share_head (`bool`, *optional*, defaults to False):
                Share the same prediction head for all exit transformer blocks. Suitable
                for ALGPT2 which shares parameters for all transformer blocks.
            exit_strategy (`str`, *optional*, defaults to "confidence"):
                The strategy to exit early. The value should be one of "confidence",
                "similarity".
                - "confidence": exit when the model is confident enough. The model will
                    exit when the probability of the predicted token is greater than the
                    threshold.
                - "similarity": exit when the model is similar enough. The model will
                    exit when the cosine similarity between the hidden state in the 
                    current layer and the hidden state in the last layer is greater than
                    the threshold.
                - "softmax": we take the difference between the top two values of softmax.
                    If the difference is greater than the threshold, we exit.
            loss_layers (`str`, *optional*, defaults to "-1"):
                The layers to calculate loss. The layers are separated by underscore. The
                default value is "-1", which means the last layer. The value "-2_-1" means
                the last two layers.
            loss_weights (`str`, *optional*, defaults to "1.0"):
                The weights of loss layers. The weights are separated by underscore. The
                default value is "1.0", which means the weight of the last layer is 1.0.
                The number of weights should be equal to the number of loss layers.
            exit_layers (`str`, *optional*, defaults to "-1"):
                The layers to exit early. The layers are separated by underscore. The
                default value is "-1", which means the last layer. The value "-2_-1" means
                the last two layers.
            exit_threshold (`float`, *optional*, defaults to 1.0):
                The threshold to exit early. The value should be in (0, 1]. The default
                value is 1.0, which means the model will not exit early.
        """
        super().__init__(**kwargs)
        self.use_sweet = use_sweet
        self.use_ln_head = use_ln_head
        self.share_head = share_head
        self.exit_strategy = exit_strategy
        self.loss_layers = loss_layers
        self.loss_weights = loss_weights
        self.exit_layers = exit_layers
        self.exit_threshold = exit_threshold

        # post init check
        loss_layers = [int(x) if int(x) >= 0 else int(x) + self.num_hidden_layers for x in self.loss_layers.split("_")]
        loss_weights = [float(x) for x in self.loss_weights.split("_")]
        exit_layers = [int(x) if int(x) >= 0 else int(x) + self.num_hidden_layers for x in self.exit_layers.split("_")]
        if len(loss_layers) != len(loss_weights):
            raise ValueError("The number of loss layers should be equal to the number of loss weights.")
        if self.exit_threshold <= 0 or self.exit_threshold > 1:
            raise ValueError("The exit threshold should be in (0, 1].")
        if (self.num_hidden_layers - 1) not in exit_layers: # XXX: is this necessary?
            raise ValueError("The last layer should be in the exit layers.")
        if (self.num_hidden_layers - 1) not in loss_layers:
            raise ValueError("The last layer should be in the loss layers.")
        
class LlamaConfig(LlamaConfigBase):
    model_type = "llama"

class ALLlamaConfig(LlamaConfigBase):
    model_type = "alllama"

class CycleLlamaConfig(LlamaConfigBase):
    model_type = "cyclellama"

    def __init__(self, cycles: int = 2, **kwargs):
        """
        Args:
            cycles (`int`, *optional*, defaults to 2):
                The number of cycles to run the model for. Each cycle will have (# of layers
                / # of cycles) Transformer blocks.
        """
        super().__init__(**kwargs)
        self.cycles = cycles
