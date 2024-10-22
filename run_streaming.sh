#!/usr/bin/env bash

# streaming test
python test_streaming.py \
    --model_name_or_path outputs/llamatiny-test \
    --output_dir streaming/llamatiny-test \
    --dataset_name pg19 \
    --download_streaming \
    --num_eval_tokens 4000000 \
    --sink_cache \
    --num_sink_tokens 4 \
    --window_length 1024 \

# performance test
python test_streaming.py \
    --model_name_or_path outputs/llamatiny-test \
    --output_dir streaming/llamatiny-2048 \
    --dataset_name pg19 \
    --download_streaming \
    --num_eval_tokens 65133 \
    --sink_cache \
    --num_sink_tokens 4 \
    --window_length 2048 \
