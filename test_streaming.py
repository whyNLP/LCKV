import torch
from tqdm import tqdm
import os
import models
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.enable_streaming_llm import enable_streaming_llm
import argparse

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

    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args

def load(model_name_or_path, torch_dtype):
    print(f"Loading model from {model_name_or_path} ...")
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

args = parse_args()

data = load_dataset(args.dataset_name, args.task, split=args.split, streaming=args.download_streaming)
if args.num_samples is not None:
    data = data.select(range(args.num_samples))

model, tokenizer = load(args.model_name_or_path, args.torch_dtype)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_streaming:
    kv_cache = enable_streaming_llm(
        model, start_size=args.start_size, recent_size=args.recent_size
    )
else:
    kv_cache = None

os.makedirs(args.output_dir, exist_ok=True)
with open(f"{args.output_dir}/log.txt", "w") as f:

    num_eval_tokens = 0
    for item in data:
        text = item['text']
        encodings = tokenizer(text, return_tensors="pt")

        print(encodings.input_ids[:, :10])

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
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
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(neg_log_likelihood.item(), file=f, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")