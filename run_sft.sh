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

accelerate launch run_sft.py \
    --model_name_or_path outputs/llamatiny-3090-test \
    --dataset_name timdettmers/openassistant-guanaco \
    --dataset_text_field text \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 2048 \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_steps 1 \
    --num_train_epochs 5 \
    --report_to none \
    --overwrite_output_dir \
    --output_dir outputs/sft_openassistant-guanaco

