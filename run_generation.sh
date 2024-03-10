# improvement: huge
export ALGPT_FLASH_ATTN=1
# improvement: significant
export ALGPT_FUSED_RMSNORM=1
# improvement: none
export ALGPT_FUSED_CROSSENTROPY=1
# improvement: none
export ALGPT_FUSED_ROTARY=1 # comment it out if num_return_sequences > 1
# improvement: slightly
export ALGPT_FUSED_SWIGLU=1

python run_generation.py \
    --model_type opt-llama \
    --torch_dtype bfloat16 \
    --model_name_or_path outputs/llamatiny-3090-test \
    --num_return_sequences 1 \
    --prompt "the meaning of life is" \
    --length 100
