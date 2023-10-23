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
""" OpenAI GPT-2 configuration"""
from transformers.utils import logging
from transformers.models.gpt2.configuration_gpt2 import GPT2Config as _GPT2Config


logger = logging.get_logger(__name__)


class GPT2ConfigBase(_GPT2Config):
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
        loss_layers = [int(x) if int(x) >= 0 else int(x) + self.n_layer for x in self.loss_layers.split("_")]
        loss_weights = [float(x) for x in self.loss_weights.split("_")]
        exit_layers = [int(x) if int(x) >= 0 else int(x) + self.n_layer for x in self.exit_layers.split("_")]
        if len(loss_layers) != len(loss_weights):
            raise ValueError("The number of loss layers should be equal to the number of loss weights.")
        if self.exit_threshold <= 0 or self.exit_threshold > 1:
            raise ValueError("The exit threshold should be in (0, 1].")
        if not set(loss_layers).issuperset(set(exit_layers)):
            raise ValueError("The exit layers should be in the loss layers.")
        if (self.n_layer - 1) not in exit_layers: # XXX: is this necessary?
            raise ValueError("The last layer should be in the exit layers.")
        if (self.n_layer - 1) not in loss_layers:
            raise ValueError("The last layer should be in the loss layers.")


class GPT2Config(GPT2ConfigBase):
    model_type = "gpt2"

class ALGPT2Config(GPT2ConfigBase):
    model_type = "algpt2"

class CycleGPT2Config(GPT2ConfigBase):
    model_type = "cyclegpt2"

    def __init__(self, cycles: int = 2, **kwargs):
        """
        Args:
            cycles (`int`, *optional*, defaults to 2):
                The number of cycles to run the model for. Each cycle will have (# of layers
                / # of cycles) GPT2 blocks.
        """
        super().__init__(**kwargs)
        self.cycles = cycles
