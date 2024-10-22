from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_lckv import LCKVLlamaConfig
from .modeling_lckv import LCKVLlamaForCausalLM, LCKVLlamaModel


AutoConfig.register("lckv-llama", LCKVLlamaConfig)
AutoModel.register(LCKVLlamaConfig, LCKVLlamaModel)
AutoModelForCausalLM.register(LCKVLlamaConfig, LCKVLlamaForCausalLM)
