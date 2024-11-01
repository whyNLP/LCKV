import argparse
from collections import defaultdict

import models
from models.utils import LayerTypeParser
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", help="The pretrained llama model to convert", required=True)
    parser.add_argument("--config_name", help="The config file of the expected LCKV model", required=True)
    parser.add_argument("--config_overrides", help="Override some existing config settings. Example: layer_types=0_6_6_6_6_6_6_7,forward_passes=7", default=None, required=False)
    parser.add_argument("--tokenizer_name", help="Pretrained tokenizer name or path if not the same as the pretrained model.", default=None, required=False)
    parser.add_argument("--output_dir", help="The output directory where the converted model will be written.", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.config_name)
    pt_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    pt_model_state_dict = pt_model.state_dict()

    assert config.model_type == "lckv-llama", "The target model must be a LCKV model"
    # allow config overrides under all circumstances
    if args.config_overrides is not None:
        print(f"Overriding config: {args.config_overrides}")
        config.update_from_string(args.config_overrides)
        print(f"New config: {config}")

    model = AutoModelForCausalLM.from_config(config)
    model_state_dict = model.state_dict()

    # Copy the weights from the pretrained model to the LCKV model
    print("Copying weights from the pretrained model to the LCKV model...")
    for name, param in pt_model.named_parameters():
        if ('k_proj' in name or 'v_proj' in name):
            continue

        if name in model_state_dict:
            model_state_dict[name].copy_(param.data)
        else:
            print(f"WARNING: {name} not found in the model")

    # Average the weights of the k_proj and v_proj layers
    # The pretrained layer weights will contribute to the layer it attends to
    # XXX: how to align heads?
    print("Averaging the weights of the k_proj and v_proj layers...")
    parser = LayerTypeParser(config.layer_types)
    k_proj, v_proj = defaultdict(list), defaultdict(list)
    for layer_type in parser:
        k_proj[layer_type.attends_to].append(pt_model_state_dict[f"model.layers.{layer_type.layer_idx}.self_attn.k_proj.weight"])
        v_proj[layer_type.attends_to].append(pt_model_state_dict[f"model.layers.{layer_type.layer_idx}.self_attn.v_proj.weight"])

    for layer_type in parser:
        if layer_type.computes_kv:
            model_state_dict[f"model.layers.{layer_type.layer_idx}.self_attn.k_proj.weight"].copy_(sum(k_proj[layer_type.layer_idx]) / len(k_proj[layer_type.layer_idx]))
            model_state_dict[f"model.layers.{layer_type.layer_idx}.self_attn.v_proj.weight"].copy_(sum(v_proj[layer_type.layer_idx]) / len(v_proj[layer_type.layer_idx]))

    # Save the model
    print(f"Saving the model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Model convertion finished successfully")

if __name__ == "__main__":
    main()
