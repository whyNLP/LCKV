from .modeling_algpt2 import ALGPT2Model, ALGPT2LMHeadModel, ALGPT2Config
from .modeling_cyclegpt2 import CycleGPT2Model, CycleGPT2LMHeadModel, CycleGPT2Config

from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
CONFIG_MAPPING.register("algpt2", ALGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(ALGPT2Config, ALGPT2LMHeadModel)
CONFIG_MAPPING.register("cyclegpt2", CycleGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(CycleGPT2Config, CycleGPT2LMHeadModel)

import os
if os.environ.get('ALGPT_FLASH_ATTN', False):
    import transformers
    from .modeling_algpt2 import GPT2Attention
    from .gpt2_flash_attention import forward
    transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = forward
    GPT2Attention.forward = forward