# Layer-Condensed KV Cache

<div align="center">
<img width="200" src="https://github.com/whyNLP/LCKV/assets/43395692/de271239-0096-4fd7-a578-59e57db916a2" />
<p>
  The KVs of the top layer
  <br>
  are the most informative and important.
  <br>
  So why bother caching the rest?
</p>
</div>

The code base for project **Layer-Condensed KV Cache**, a new variant of transformer decoders in which queries of all layers are paired with keys and values of just the top layer. It reduces the memory and computation cost, reduces the number of parameters, significantly improves the inference throughput with comparable or better task performance. The paper "[Layer-Condensed KV Cache for Efficient Inference of Large Language Models](https://arxiv.org/abs/2405.10637)" was accepted to ACL 2024 main conference.

This work is inspired by [Probabilistic Transformer](https://github.com/whyNLP/Probabilistic-Transformer), where we consider the stacking layer structure of a transformer as an iterative process of improving token representation.

<details>
<summary>The Map of AI Approaches</summary>
<div align="center">
<img width="400" src="https://github.com/whyNLP/LCKV/assets/43395692/cdca6717-8a30-4e24-9b61-c8ad743bc092" />
</div>
</details>

## News

> [!WARNING]
> This branch is under active development. The code may not work in the way exactly the same as that in the paper. Please refer to the [published branch](https://github.com/whyNLP/LCKV/tree/dev-lckv-publish) for reproduction purposes.

- [24/10/18] Our new empirical study "[A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference](http://arxiv.org/abs/2410.14442)" has released on arXiv. A new configuration has been found to be more efficient than the original LCKV.
- [24/05/28] This code base now also supports Cross-Layer Attention (CLA). The idea is similar, but they 1) divide the transformer layers into small groups with 2-4 layers in each group; 2) pairs the queries of all the layers with the keys and values of the bottom layer in each group. See details in their paper "[Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](http://arxiv.org/abs/2405.12981)".
- [24/05/20] LCKV initial code release. 

## Installation

You may install the dependencies with the following commands:

```sh
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

where the CUDA version is set to `12.1`. For other CUDA versions, please refer to installation instructions of [PyTorch](https://pytorch.org/get-started/locally/). See [Trouble shooting](#trouble-shooting) for more details.

## Usage

Our implementation is based on HuggingFace `transformers`. We register a new model `lckv-llama` that supports the Layer-Condensed KV Cache. It inherits from the `llama` model and adds support for the Layer-Condensed KV Cache.

> [!NOTE]
> It is difficult to support the Layer-Condensed KV Cache for a variety of models with a small amount of code. This is because the Layer-Condensed KV Cache requires to modify the attention mechanism and training recipe of the transformer decoder. Currently, we only implemented the Layer-Condensed KV Cache for the `llama` model, and it is possible to extend it to other models with similar structures.

```python
import models # register the lckv-llama model
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_config(config="configs/tinyllama_lckv.json")
```

and now you have a randomly initialized model with the Layer-Condensed KV Cache.

### Optimization

To accelerate the training and inference of the model, one could apply the liger kernel supported by `transformers` library. The provided training script `run_clm.py` has already activated the liger kernel. See more details [here](https://huggingface.co/docs/transformers/v4.45.2/en/trainer#liger-kernel).

### Configuration

We provide some sample configuration files in the  `configs` folder. The config settings are defined in [models/configuration_lckv.py](models/configuration_lckv.py). You may refer to this file for more details.

#### Option 1: Modify the configurations in python:

```python
from models import LCKVLlamaConfig

# we have prepared a sample configuration file
config = LCKVLlamaConfig.from_pretrained("configs/tinyllama_lckv.json")

# below is the LCKV config. you may modify the configuration as you like
config.forward_passes  = 7      # m in the paper
config.backward_passes = 2      # b in the paper
config.layer_types     = "0_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_21" # for each layer, which layer to attend to

# we also support this
config.layer_types     = "0_10_10_10_10_10_10_10_10_10_10_10_10_10_10_10_10_10_10_10_10_21" # the sandwich-middle configuration
config.layer_types     = "0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21" # Llama config
config.layer_types     = "0_0_2_2_4_4_6_6_8_8_10_10_12_12_14_14_16_16_18_18_20_20" # CLA config

config.sliding_window  = 1024   # the window size for the sliding window attention
config.layer_types     = "0s_1s_2s_3s_4s_5s_6s_7s_8s_9s_10s_11_11_11_11_11_11_11_11_11_11_11" # YOCO config, 's' is for sliding window

config.sliding_window  = 1024   # the window size for the sliding window attention
config.layer_types     = "0_1s_1s_3s_3s_3s_0_7s_7s_9s_9s_9s_12_13s_13s_15s_15s_15s_12_19s_19s_19s" # MixAttention (Pairs) config
```

#### Option 2: Modify the configurations in the shell script (via `--config_overrides`):

```sh
accelerate launch run_clm.py \
    --config_name configs/tinyllama_lckv.json \
    --config_overrides forward_passes=7,backward_passes=2,layer_types=0_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_20_21 \
    ...
```

With the above configurations, you can create [CLA](http://arxiv.org/abs/2405.12981), [YOCO](https://arxiv.org/abs/2405.05254) or any configurations in [Cross-Layer KV Sharing](http://arxiv.org/abs/2410.14442) or [MixAttention](http://arxiv.org/abs/2409.15012) without changing the code. The only thing you need to do is to write the correct `layer_types` in the configuration file.

### Training

We use the same [training script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) as the original `transformers` library. You may refer to the [official documentation](https://huggingface.co/transformers/training.html) for more details.

We provide a training script `run_clm.sh` for training a 50M parameter model on the `wikitext-103` dataset. You may run the script with:

```sh
bash run_clm.sh
```

See the script for more details. For pretraining on SlimPajama, please follow the instructions in [tinyllama-zh](https://github.com/whyNLP/tinyllama-zh) and replace the dataset with SlimPajama.

### Inference

We use the same [inference script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) as the original `transformers` library. To perform inference, you may run the following command:

```sh
bash run_generation.sh
```

You may get responses from the trained model given any prompts. See the script for more details.

#### Streaming

We integrate our model with [StreamingLLM](https://github.com/mit-han-lab/streaming-llm). To perform streaming inference, you may run the following command:

```sh
bash run_streaming.sh
```

See the script for more details. The `run_generation.py` script also supports streaming inference with the `--sink_cache` flag.

#### Sliding Window Attention

The generation script also supports sliding window attention inference. If the model is trained with sliding window attention, the generation script will automatically use the sliding window attention for inference.

### Evaluation

We use [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the model. You may run the following command:

```sh
python test_harness.py --model_name_or_path ...
```

with the path to the model checkpoint. Run `python test_harness.py --help` for more details.

### Latency Testing

To test the latency of the model, you may run the following command:

```sh
python test_latency.py
```


## Trouble shooting

### Flash-Attn Installation

https://github.com/Dao-AILab/flash-attention/issues/451

Behavior:

Runtime error.
```sh
ImportError: /home/.../flash_attn_2_cuda.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops9_pad_enum4callERKNS_6TensorEN3c108ArrayRefINS5_6SymIntEEElNS5_...
```

Solution:
```sh
pip uninstall flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```

### CUDA version

The cuda version may affect the installation of:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Flash-Attn](https://github.com/Dao-AILab/flash-attention)

Please make sure to install the correct version of the packages (so long as they are consistent, the code would work). Also make sure that `nvcc` is installed and available in the path.

Our experiment environment uses `CUDA 12.1` and you may install with
```sh
conda install pytorch==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Questions

> 1. Is it possible to integrate the LCKV with MQA / GQA?

Yes. The fact is that we have already done this in our experiments. Tinyllama uses 32 attention heads and 4 KV heads. We follow the same setting in our experiments. If you want to experiment with different settings, you may modify the `num_attention_heads` and `num_key_value_heads` in the configuration file.
