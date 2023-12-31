"""
This script will convert a proj_kv model to a use_kv model, by fusing the projection parameters into W_Q and W_O.
"""

from dataclasses import dataclass, field
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from models.modeling_llama import LlamaAttentionProj

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    input: str = field(
        default=None,
        metadata={
            "help": (
                "The input model path."
            )
        },
    )

    output: str = field(
        default=None,
        metadata={
            "help": (
                "The output model path."
            )
        },
    )
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments,))
    (model_args, ) = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.input)
    assert config.model_type == "kv-llama"

    tokenizer = AutoTokenizer.from_pretrained(model_args.input)

    model = AutoModelForCausalLM.from_pretrained(model_args.input, config=config)

    # fuse the parameters
    for layer in model.model.layers:
        module = layer.self_attn
        if isinstance(module, LlamaAttentionProj):
            for i in range(module.num_key_value_heads):
                # self.head_dim, self.num_key_value_heads * self.head_dim
                # self.hidden_size, self.num_heads * self.head_dim
                k_proj = module._last_k_proj.weight.data[i*module.head_dim:(i+1)*module.head_dim]
                v_proj = module._last_v_proj.weight.data[i*module.head_dim:(i+1)*module.head_dim]
                n_rep = module.num_heads // module.num_key_value_heads
                for j in range(i*n_rep, (i+1)*n_rep):
                    module.q_proj.weight.data[j*module.head_dim:(j+1)*module.head_dim] = k_proj @ module.q_proj.weight.data[j*module.head_dim:(j+1)*module.head_dim]
                    module.o_proj.weight.data[:,j*module.head_dim:(j+1)*module.head_dim] = module.o_proj.weight.data[:,j*module.head_dim:(j+1)*module.head_dim] @ v_proj.T
            del module._last_k_proj
            del module._last_v_proj
    
    config.update_from_string("kv_pattern=use_kv")

    model.save_pretrained(model_args.output)
    tokenizer.save_pretrained(model_args.output)
    config.save_pretrained(model_args.output)

if __name__ == '__main__':
    main()