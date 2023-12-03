from .configuration_gpt2 import GPT2Config, ALGPT2Config, CycleGPT2Config
from .modeling_gpt2 import GPT2LMHeadModel, ALGPT2LMHeadModel, CycleGPT2LMHeadModel
from transformers.models.llama.configuration_llama import LlamaConfig
from .modeling_llama import LlamaForCausalLM
from .wandb_callback import WandbCallback

from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
CONFIG_MAPPING.register("gpt2", GPT2Config, exist_ok = True)
MODEL_FOR_CAUSAL_LM_MAPPING.register(GPT2Config, GPT2LMHeadModel, exist_ok = True)
CONFIG_MAPPING.register("algpt2", ALGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(ALGPT2Config, ALGPT2LMHeadModel)
CONFIG_MAPPING.register("cyclegpt2", CycleGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(CycleGPT2Config, CycleGPT2LMHeadModel)

MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM, exist_ok = True)

import os
if os.environ.get('ALGPT_FLASH_ATTN', False):
    import transformers
    from .gpt2_flash_attention import forward
    transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = forward

if os.environ.get('ALGPT_FUSED_RMSNORM', False):
    import transformers
    from flash_attn.ops.rms_norm import RMSNorm
    transformers.models.llama.modeling_llama.LlamaRMSNorm = RMSNorm