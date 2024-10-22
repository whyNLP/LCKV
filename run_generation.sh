#!/usr/bin/env bash

python run_generation.py \
    --model_type lckv-llama \
    --torch_dtype bfloat16 \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --model_name_or_path outputs/test-lckv \
    --num_return_sequences 1 \
    --prompt "the meaning of life is" \
    --length 100 \
    --sink_cache \
    --window_length 20
