import models
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the pretrained llama model to convert")
    parser.add_argument("--config", help="the config file of the expected LCKV model")
    parser.add_argument("--config_overrides", help="overrides for the config file", default=None, required=False)
    parser.add_argument("--tokenizer", help="path to the tokenizer", default=None, required=False)
    parser.add_argument("--average_all_kvs", help="average all kvs, otherwise only average kvs required", action="store_true")
    parser.add_argument("--output", help="path to the output model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.input)
    config = AutoConfig.from_pretrained(args.config)
    pt_model = AutoModelForCausalLM.from_pretrained(args.input)

    assert config.model_type == "opt-llama", "The target model must be a LCKV model"
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
    
        print(name, param.data.size())

    # Average the weights of the k_proj and v_proj layers
    # XXX: how to align heads?
    print("Averaging the weights of the k_proj and v_proj layers...")
    layer_types = [int(x) for x in config.layer_types.split("_")]
    k_proj_all = 0
    v_proj_all = 0
    total_layers = 0
    for i, tp in enumerate(layer_types):
        k_proj = pt_model.state_dict()[f"model.layers.{i}.self_attn.k_proj.weight"]
        v_proj = pt_model.state_dict()[f"model.layers.{i}.self_attn.v_proj.weight"]

        if tp == 0:
            model_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"].copy_(k_proj)
            model_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"].copy_(v_proj)
            if args.average_all_kvs:
                k_proj_all += k_proj
                v_proj_all += v_proj
                total_layers += 1
        else:
            k_proj_all += k_proj
            v_proj_all += v_proj
            total_layers += 1
    
    target_layer = config.target_layer % config.num_hidden_layers
    model_state_dict[f"model.layers.{target_layer}.self_attn.k_proj.weight"].copy_(k_proj_all / total_layers)
    model_state_dict[f"model.layers.{target_layer}.self_attn.v_proj.weight"].copy_(v_proj_all / total_layers)

    # Save the model
    print("Saving the model...")
    model.save_pretrained(args.output)
    config.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("Model convertion finished successfully")

if __name__ == "__main__":
    main()