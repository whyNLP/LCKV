#!/usr/bin/env bash

## pretrain code for llama-tiny
#  - to pretrain a tinyllama, change the config to `TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T`
#  - to intialize the model with a pretrained model, add `--model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T`
#  - to use the minipile dataset, use `--dataset_name JeanKaddour/minipile`, with proper `--preprocessing_num_workers`
#  - to enable wandb, use `--report_to wandb`
accelerate launch run_clm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/llama_tiny_lckv.json \
    --config_overrides layer_types=0_6_6_6_6_6_6_7,forward_passes=7,backward_passes=2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --auto_find_batch_size \
    --gradient_accumulation_steps 1 \
    --block_size 1024 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 3e-4 \
    --weight_decay 1e-1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --run_name llamatiny-test \
    --overwrite_output_dir \
    --output_dir outputs/llamatiny-test


## eval code for llama-tiny
python run_clm.py \
    --model_name_or_path outputs/llamatiny-test \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_eval_batch_size 8 \
    --block_size 1024 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_eval \
    --report_to none \
    --overwrite_output_dir \
    --output_dir /tmp/test-clm-`date +"%m-%d--%H-%M-%S"`-$RANDOM
