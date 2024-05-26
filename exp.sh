
export WANDB_PROJECT=Llama-test
echo "Start running..."
echo "Slurm job id: $SLURM_JOB_ID"

# improvement: huge
export LCKV_FLASH_ATTN=1
# improvement: significant
export LCKV_FUSED_RMSNORM=1
# improvement: none
export LCKV_FUSED_CROSSENTROPY=1
# improvement: none
export LCKV_FUSED_ROTARY=1
# improvement: slightly
export LCKV_FUSED_SWIGLU=1

## pretrain code for llama-tiny
#  - to pretrain a tinyllama, change the config to `TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T`
#  - to intialize the model with a pretrained model, add `--model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T`
#  - to use the minipile dataset, use `--dataset_name JeanKaddour/minipile`, with proper `--preprocessing_num_workers`
#  - to enable wandb, use `--report_to wandb`
export TRANSFORMERS_OFFLINE=1
accelerate launch run_clm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/tinyllama_opt.json \
    --config_overrides num_encoders=8,num_trained_encoders=1,layer_types=0_0_0_0_0_1_1_1_1_1_1_2_1_1_1_1_1_0_0_0_0_0,target_layer=11,train_kv=false \
    --dataset_name JeanKaddour/minipile \
    --preprocessing_num_workers 96 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 3e-4 \
    --weight_decay 6.6e-6 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --run_name opt-tinyllama-pt-w10-sandwich-cache-middle \
    --output_dir outputs/opt-tinyllama-pt-w10-sandwich-cache-middle


## uncomment to enable evaluation at token level (each token will only look at the last layer of the previous tokens)
# export LCKV_INFERENCE=1

# ## eval code for llama-tiny
# python run_clm.py \
#     --model_name_or_path outputs/llamatiny-3090-test \
#     --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_eval_batch_size 8 \
#     --block_size 1024 \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --do_eval \
#     --report_to none \
#     --overwrite_output_dir \
#     --output_dir /tmp/test-clm-`date +"%m-%d--%H-%M-%S"`-$RANDOM
