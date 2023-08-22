
echo "Start running..."
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --save_total_limit 3 \
    --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`


# --model_name_or_path huggyllama/llama-7b \
# --quantization 4bit \