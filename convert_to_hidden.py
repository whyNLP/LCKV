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
from models.configuration_llama import BestLlamaConfig

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
    assert config.model_type == "hidden-llama"

    tokenizer = AutoTokenizer.from_pretrained(model_args.input)

    model = AutoModelForCausalLM.from_pretrained(model_args.input, config=config)

    model.model = model.tgt_model
    model.lm_head = model.tgt_lm_head
    del model.tgt_model
    del model.tgt_lm_head
    
    config = BestLlamaConfig.from_pretrained(model_args.input)
    del config.loss_weights
    del config.target_type

    model.save_pretrained(model_args.output)
    tokenizer.save_pretrained(model_args.output)
    config.save_pretrained(model_args.output)

if __name__ == '__main__':
    main()