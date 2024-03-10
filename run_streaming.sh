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

# run_streaming.py is not supported yet

# streaming test
python test_streaming.py \
    --model_name_or_path outputs/llamatiny-3090-test \
    --output_dir streaming/llamatiny-3090-test \
    --dataset_name pg19 \
    --num_eval_tokens 4000000 \
    --enable_streaming \
    --start_size 4 \
    --recent_size 1020 \

# performance test
python test_streaming.py \
    --model_name_or_path outputs/llamatiny-3090-test \
    --output_dir streaming/llamatiny-2048 \
    --dataset_name pg19 \
    --download_streaming \
    --num_eval_tokens 65133 \
    --enable_streaming \
    --start_size 4 \
    --recent_size 2044 \
