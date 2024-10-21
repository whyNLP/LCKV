from .configuration_lckv import LCKVLlamaConfig
from .modeling_lckv import LCKVLlamaModel, LCKVLlamaForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
AutoConfig.register("lckv-llama", LCKVLlamaConfig)
AutoModel.register(LCKVLlamaConfig, LCKVLlamaModel)
AutoModelForCausalLM.register(LCKVLlamaConfig, LCKVLlamaForCausalLM)
