from .configuration_gpt2 import GPT2Config, ALGPT2Config, CycleGPT2Config
from .modeling_gpt2 import GPT2LMHeadModel, ALGPT2LMHeadModel, CycleGPT2LMHeadModel
from .configuration_llama import BestLlamaConfig
from .modeling_llama import LlamaForCausalLM
from .configuration_llama import ALLlamaConfig, CycleLlamaConfig
from .modeling_alllama import ALLlamaForCausalLM, CycleLlamaForCausalLM
from .configuration_llama import KVLlamaConfig, HiddenLlamaConfig
from .modeling_llamakv import LlamaKVForCausalLM, LlamaHiddenForCausalLM
from .wandb_callback import WandbCallback

from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
CONFIG_MAPPING.register("gpt2", GPT2Config, exist_ok = True)
MODEL_FOR_CAUSAL_LM_MAPPING.register(GPT2Config, GPT2LMHeadModel, exist_ok = True)
CONFIG_MAPPING.register("algpt2", ALGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(ALGPT2Config, ALGPT2LMHeadModel)
CONFIG_MAPPING.register("cyclegpt2", CycleGPT2Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(CycleGPT2Config, CycleGPT2LMHeadModel)
CONFIG_MAPPING.register("best-llama", BestLlamaConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(BestLlamaConfig, LlamaForCausalLM)

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("alllama", ALLlamaConfig)
AutoModelForCausalLM.register(ALLlamaConfig, ALLlamaForCausalLM)
AutoConfig.register("cyclellama", CycleLlamaConfig)
AutoModelForCausalLM.register(CycleLlamaConfig, CycleLlamaForCausalLM)
AutoConfig.register("kv-llama", KVLlamaConfig)
AutoModelForCausalLM.register(KVLlamaConfig, LlamaKVForCausalLM)
AutoConfig.register("hidden-llama", HiddenLlamaConfig)
AutoModelForCausalLM.register(HiddenLlamaConfig, LlamaHiddenForCausalLM)

import os
if os.environ.get('ALGPT_FLASH_ATTN', False):
    import transformers
    from .gpt2_flash_attention import forward
    transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = forward

if os.environ.get('ALGPT_FUSED_RMSNORM', False):
    import transformers
    from flash_attn.ops.rms_norm import RMSNorm
    transformers.models.llama.modeling_llama.LlamaRMSNorm = RMSNorm
    from . import modeling_llama
    modeling_llama.LlamaRMSNorm = RMSNorm
    from . import modeling_alllama
    modeling_alllama.LlamaRMSNorm = RMSNorm

if os.environ.get('ALGPT_FUSED_CROSSENTROPY', False):
    import transformers
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    transformers.models.llama.modeling_llama.CrossEntropyLoss = CrossEntropyLoss
    from . import modeling_llama
    modeling_llama.CrossEntropyLoss = CrossEntropyLoss
    from . import modeling_alllama
    modeling_alllama.CrossEntropyLoss = CrossEntropyLoss

if os.environ.get('ALGPT_FUSED_ROTARY', False):
    import transformers
    from .llama_fused_rotary import (
        LlamaRotaryEmbedding,
        LlamaLinearScalingRotaryEmbedding,
        LlamaDynamicNTKScalingRotaryEmbedding,
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_q
    )
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = LlamaLinearScalingRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbedding

    from . import modeling_llama
    modeling_llama.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    modeling_llama.apply_rotary_pos_emb_q = fused_apply_rotary_pos_emb_q

    from . import modeling_llamakv
    modeling_llamakv.apply_rotary_pos_emb_q = fused_apply_rotary_pos_emb_q

if os.environ.get('ALGPT_FUSED_SWIGLU', False):
    import transformers
    from .llama_fused_swiglu import LlamaMLP
    from . import modeling_llama
    modeling_llama.LlamaMLP = LlamaMLP
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP