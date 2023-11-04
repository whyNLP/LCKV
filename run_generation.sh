# python run_generation.py \
#     --model_type=origin-gpt2 \
#     --model_name_or_path=gpt2 \
#     --num_return_sequences 10

python run_generation.py \
    --model_type algpt2 \
    --model_name_or_path /home/wuhy/projects/algpt/ALGPT/outputs/algpt2-nlayer24 \
    --config_overrides exit_layers=4_8_12_16_20_23,exit_threshold=0.99928,n_layer=24,exit_strategy=similarity \
    --num_return_sequences 1 \
    --length 100