from .configuration_llama import OptLlamaConfig
from .modeling_llama_opt import LlamaForCausalLM as OptLlamaForCausalLM
from .wandb_callback import WandbCallback

from .modeling_llama_cla import LlamaForCausalLM as ClaLlamaForCausalLM
from .configuration_llama import ClaLlamaConfig

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("opt-llama", OptLlamaConfig)
AutoModelForCausalLM.register(OptLlamaConfig, OptLlamaForCausalLM)
AutoConfig.register("cla-llama", ClaLlamaConfig)
AutoModelForCausalLM.register(ClaLlamaConfig, ClaLlamaForCausalLM)

import os

if os.environ.get('LCKV_FUSED_RMSNORM', False):
    import transformers
    from flash_attn.ops.rms_norm import RMSNorm
    transformers.models.llama.modeling_llama.LlamaRMSNorm = RMSNorm
    from . import modeling_llama_opt
    modeling_llama_opt.LlamaRMSNorm = RMSNorm
    from . import modeling_llama_cla
    modeling_llama_cla.LlamaRMSNorm = RMSNorm

if os.environ.get('LCKV_FUSED_CROSSENTROPY', False):
    import transformers
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    transformers.models.llama.modeling_llama.CrossEntropyLoss = CrossEntropyLoss
    from . import modeling_llama_opt
    modeling_llama_opt.CrossEntropyLoss = CrossEntropyLoss
    from . import modeling_llama_cla
    modeling_llama_cla.CrossEntropyLoss = CrossEntropyLoss

if os.environ.get('LCKV_FUSED_ROTARY', False):
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

    from . import modeling_llama_opt
    modeling_llama_opt.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    modeling_llama_opt.apply_rotary_pos_emb_q = fused_apply_rotary_pos_emb_q

    from . import modeling_llama_opt_streaming
    modeling_llama_opt_streaming.apply_rotary_pos_emb_q = fused_apply_rotary_pos_emb_q

    from . import modeling_llama_cla
    modeling_llama_cla.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    modeling_llama_cla.apply_rotary_pos_emb_q = fused_apply_rotary_pos_emb_q

if os.environ.get('LCKV_FUSED_SWIGLU', False):
    import transformers
    from .llama_fused_swiglu import LlamaMLP
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    from . import modeling_llama_opt
    modeling_llama_opt.LlamaMLP = LlamaMLP
    from . import modeling_llama_cla
    modeling_llama_cla.LlamaMLP = LlamaMLP

try:
    from streaming_llm import enable_streaming_llm
    from .modeling_llama_opt_streaming import enable_streaming_llm as custom_enable_streaming_llm
    enable_streaming_llm.enable_streaming_llm = custom_enable_streaming_llm

    if os.environ.get('LCKV_FUSED_ROTARY', False):
        from .llama_fused_rotary import fused_apply_rotary_pos_emb_q
        from streaming_llm.pos_shift import modify_llama
        modify_llama.apply_rotary_pos_emb_single = fused_apply_rotary_pos_emb_q
except:
    pass
