# Full training
python run_sft.py \
    --model_name_or_path outputs/llamatiny-test \
    --dataset_name timdettmers/openassistant-guanaco \
    --dataset_text_field text \
    --max_seq_length 1024 \
    --use_liger \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.41e-5 \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --report_to none \
    --output_dir outputs/llamatiny-sft-test
