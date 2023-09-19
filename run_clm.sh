
export WANDB_PROJECT=ALGPT2-Tuning
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


## When using flash attention, the memory usage is reduced.
export ALGPT_FLASH_ATTN=1
## gpt2-small     (bz64)  66G
## algpt2-tiny    (bz128) 73G
## algpt2-small   (bz16)  17G
## algpt2-small   (bz64)  64G
## algpt2-medium  (bz32)  65G
## algpt2-large   (bz16)  56G
## algpt2-xl      (bz8)   46G
## algpt2-xl      (bz12)  67G
## cyclegpt2-tiny (bz64)  44G


# # gpt2 deepspeed
# deepspeed --master_port 60000 run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type gpt2 \
#     --config_name configs/gpt/config_tiny.json \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 5e-4 \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 30 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to wandb \
#     --run_name gpt2-tiny \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

# # algpt2 deepspeed
# deepspeed --master_port 60003 run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type algpt2 \
#     --config_name configs/algpt/config_tiny.json \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 128 \
#     --per_device_eval_batch_size 128 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.006 \
#     --learning_rate 2e-3 \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 100 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to wandb \
#     --run_name algpt2-tiny-long \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

# cyclegpt2 deepspeed
deepspeed --master_port 60001 run_clm.py \
    --deepspeed ds_config.json \
    --model_type cyclegpt2 \
    --config_name configs/cyclegpt/config_tiny.json \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --learning_rate 1e-3 \
    --bf16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 30 \
    --save_total_limit 3 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --run_name cyclegpt2-tiny-c2-1e-3 \
    --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`
