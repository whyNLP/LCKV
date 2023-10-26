
export WANDB_PROJECT=ALGPT2-Tuning
echo "Start running..."
echo "Slurm job id: $SLURM_JOB_ID"

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}

MASTER_PORT=$(rand 50000 60000)


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

## algpt2-tiny    (head=8, bz48) 44G


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
# deepspeed --master_port $MASTER_PORT run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type algpt2 \
#     --config_name configs/algpt/config_tiny.json \
#     --config_overrides n_layer=8 \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 2e-3 \
#     --weight_decay 1e-5 \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 30 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to wandb \
#     --run_name algpt2-nlayer8 \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`
#     # --output_dir /home/wuhy/projects/algpt/ALGPT/outputs/algpt2-nlayer8

# # cyclegpt2 deepspeed
# deepspeed --master_port $MASTER_PORT run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type cyclegpt2 \
#     --config_name configs/cyclegpt/config_tiny.json \
#     --tokenizer_name gpt2 \
#     --config_overrides loss_layers=15_23,loss_weights=0.2_1,cycles=3,n_layer=24 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 2e-3 \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 30 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to wandb \
#     --run_name cyclegpt2-tiny-2loss-l24 \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`


# ==================================================================================================
# # algpt2 deepspeed TEST
# deepspeed --master_port $MASTER_PORT run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type algpt2 \
#     --config_name configs/algpt/config_tiny.json \
#     --config_overrides loss_layers=-1,loss_weights=1,exit_layers=1_2_3_4_5_6_7,exit_threshold=0.99,use_sweet=false,n_layer=8,exit_strategy=similarity \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 2e-3 \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --max_steps 30 \
#     --save_total_limit 3 \
#     --save_strategy steps \
#     --evaluation_strategy steps \
#     --eval_steps 10 \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`
#     # --run_name tmp-algpt2 \
#     # --config_overrides loss_layers=-1,loss_weights=1,exit_layers=1_2_3_4_5_6_7,exit_threshold=0.99,use_sweet=false,n_layer=8,exit_strategy=confidence \


# # algpt2 deepspeed
# deepspeed --master_port $MASTER_PORT run_clm.py \
#     --deepspeed ds_config.json \
#     --model_type algpt2 \
#     --config_name configs/algpt/config_tiny.json \
#     --config_overrides loss_layers=-1,loss_weights=1,exit_layers=1_2_3_4_5_6_7,exit_threshold=0.99,use_sweet=false,n_layer=8,exit_strategy=similarity \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 2e-3 \
#     --bf16 \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --num_train_epochs 30 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`
#     # --run_name tmp-algpt2 \

# # algpt2 inference
# python run_clm.py \
#     --model_name_or_path /home/wuhy/projects/algpt/ALGPT/outputs/algpt2-nlayer24 \
#     --config_name /home/wuhy/projects/algpt/ALGPT/outputs/algpt2-nlayer24 \
#     --config_overrides n_layer=4 \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 2e-3 \
#     --bf16 \
#     --do_eval \
#     --max_steps 30 \
#     --save_total_limit 3 \
#     --save_strategy steps \
#     --evaluation_strategy steps \
#     --eval_steps 10 \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`

# # cyclegpt2 inference
# python run_clm.py \
#     --model_name_or_path /tmp/test-clm-5581-10-25--19-12-26 \
#     --config_name /tmp/test-clm-5581-10-25--19-12-26 \
#     --tokenizer_name gpt2 \
#     --config_overrides loss_layers=-1,loss_weights=1,exit_layers=-1,cycles=5,n_layer=40 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.02 \
#     --learning_rate 2e-3 \
#     --bf16 \
#     --do_eval \
#     --num_train_epochs 30 \
#     --save_total_limit 3 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`