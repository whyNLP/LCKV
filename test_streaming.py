import argparse
import os

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

import models
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import SinkCache


device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )
    parser.add_argument(
        "--download_streaming", action="store_true", help="enable streaming download"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="streaming/debug",
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
    )

    # sink cache related arguments
    parser.add_argument("--sink_cache", action="store_true", help="Whether to use sink cache.")
    parser.add_argument("--window_length", type=int, default=256, help="Window size for sink cache.")
    parser.add_argument("--num_sink_tokens", type=int, default=2, help="Number of sink tokens.")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args

def load(model_name_or_path, torch_dtype):
    print(f"Loading model from {model_name_or_path} ...")
    # if only model type is specified, load from scratch
    if ";" in model_name_or_path:
        from test_latency import prepare
        tokenizer, model = prepare(*model_name_or_path.split(";"))
        return model, tokenizer
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    torch_dtype = (
        torch_dtype
        if torch_dtype in ["auto", None]
        else getattr(torch, torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model.eval()
    return model, tokenizer

def main():

    args = parse_args()

    data = load_dataset(args.dataset_name, args.task, split=args.split, streaming=args.download_streaming)
    if args.num_samples is not None:
        data = data.select(range(args.num_samples))

    model, tokenizer = load(args.model_name_or_path, args.torch_dtype)

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")

    # streaming inference
    past_key_values = None
    if args.sink_cache:
        past_key_values = SinkCache(args.window_length, args.num_sink_tokens)

    ## uncomment the following lines to enable latency measurement
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/log.txt", "w") as f:

        num_eval_tokens = 0
        for item in data:
            text = item['text']
            encodings = tokenizer(text, return_tensors="pt")

            print(encodings.input_ids[:, :10])

            seq_len = encodings.input_ids.size(1)
            print(f"num_eval_tokens: {num_eval_tokens}, seq_len: {seq_len}")
            pbar = tqdm(range(0, seq_len - 1))

            # import time
            for idx in pbar:
                # if idx == args.start_size + args.recent_size:
                #     print("Starting timer...")
                #     start = time.time()
                input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                    past_key_values = outputs.past_key_values
                    label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                    neg_log_likelihood = loss_fn(logits, label)
                nlls.append(neg_log_likelihood)
                pbar.set_description(
                    f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
                )
                print(neg_log_likelihood.item(), file=f, flush=True)
                num_eval_tokens += 1
                if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                    # print(f"time: {time.time() - start:.2f}")
                    break
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl.item())
    with open(f"{args.output_dir}/ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")

if __name__ == "__main__":
    main()
