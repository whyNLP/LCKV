from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.utils import is_liger_kernel_available

from .configuration_lckv import LCKVLlamaConfig
from .modeling_lckv import LCKVLlamaForCausalLM, LCKVLlamaModel


AutoConfig.register("lckv-llama", LCKVLlamaConfig)
AutoModel.register(LCKVLlamaConfig, LCKVLlamaModel)
AutoModelForCausalLM.register(LCKVLlamaConfig, LCKVLlamaForCausalLM)


if is_liger_kernel_available():
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    from .kernel import apply_liger_kernel_to_lckv_llama
    MODEL_TYPE_TO_APPLY_LIGER_FN["lckv-llama"] = apply_liger_kernel_to_lckv_llama
