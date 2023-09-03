
export WANDB_PROJECT=ALGPT2
echo "Start running..."

# # gpt2
# python run_clm.py \
#     --model_type gpt2 \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 3 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to wandb \
#     --run_name gpt2 \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

# # algpt2
# python run_clm.py \
#     --model_type algpt2 \
#     --tokenizer_name gpt2 \
#     --config_name configs/config.json \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 3 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to wandb \
#     --run_name algpt2 \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`


## When using deepspeed, remember to change batch size
## gpt2-small    (bz2)  11G
## gpt2-small    (bz4)  17G
## gpt2-small    (bz8)  30G
## gpt2-small    (bz16) 54G
## algpt2-medium (bz2)  18G
## algpt2-medium (bz8)  60G
## algpt2-large  (bz2)  30G


# export ALGPT_TORCH_ATTN=1
export ALGPT_FLASH_ATTN=1

# # gpt2 deepspeed
# deepspeed --master_port 60008 run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type gpt2 \
#     --config_name gpt2 \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --learning_rate 5e-4 \
#     --do_train \
#     --do_eval \
#     --max_steps 100000 \
#     --save_total_limit 3 \
#     --save_strategy steps \
#     --save_steps 2000 \
#     --evaluation_strategy steps \
#     --eval_steps 2000 \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

# algpt2 deepspeed
deepspeed --master_port 60001 run_clm.py \
    --deepspeed ds_config.json \
    --model_type algpt2 \
    --config_name configs/config_medium.json \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-4 \
    --bf16 \
    --do_train \
    --do_eval \
    --max_steps 100000 \
    --save_total_limit 3 \
    --save_strategy steps \
    --save_steps 2000 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

