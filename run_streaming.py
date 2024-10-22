import argparse
import json
import os
from pathlib import Path

import requests
import torch

import models
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from transformers.cache_utils import SinkCache
from transformers.generation.streamers import TextStreamer


logging.get_logger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.get_logger("transformers.generation.utils").setLevel(logging.ERROR)
logging.get_logger("transformers.models.llama.modeling_llama").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--data_root", type=str, default="data/")

    # generation related arguments
    parser.add_argument("--length", type=int, default=1000)

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    # sink cache related arguments
    parser.add_argument("--sink_cache", action="store_true", help="Whether to use sink cache.")
    parser.add_argument("--window_length", type=int, default=256, help="Window size for sink cache.")
    parser.add_argument("--num_sink_tokens", type=int, default=2, help="Number of sink tokens.")

    args = parser.parse_args()

    # load model
    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        chat_template=r"""
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- 'USER: ' + message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- 'ASSISTANT: '  + message['content'] + ' \n\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- 'ASSISTANT: ' }}
{%- endif %}
"""
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # load data
    print(f"Loading data from {args.data_root} ...")
    mt_bench = Path(args.data_root) / "mt_bench.jsonl"
    if not mt_bench.exists():
        print("Downloading mt_bench data ...")
        os.makedirs(args.data_root, exist_ok=True)
        with open(mt_bench, "w") as f:
            url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
            response = requests.get(url)
            f.write(response.text)

    prompts = []
    with open(mt_bench, "r") as f:
        for line in f:
            prompts += json.loads(line)["turns"]

    # streaming inference
    kwargs = {}
    if args.sink_cache:
        kwargs["past_key_values"] = SinkCache(args.window_length, args.num_sink_tokens)

    chat_history = []
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    for prompt in prompts:
        new_prompt = {"role": "user", "content": prompt}
        print(tokenizer.apply_chat_template([new_prompt], add_generation_prompt=True, tokenize=False), end="")

        chat_history.append(new_prompt)
        input_ids = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, return_tensors="pt").to(model.device)

        output_sequences = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.length,
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            streamer=streamer,
            **kwargs,
        )

        chat_history.append({"role": "assistant", "content": tokenizer.decode(output_sequences[0, input_ids.shape[-1]:], skip_special_tokens=True)})
        print()


if __name__ == "__main__":
    main()
