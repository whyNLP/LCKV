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
- [24/05/28] This code base now also supports Cross-Layer Attention (CLA). The idea is similar, but they 1) devide the transformer layers into small groups with 2-4 layers in each group; 2) pairs the queries of all the layers with the keys and values of the bottom layer in each group. See details in their paper "[Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](http://arxiv.org/abs/2405.12981)".
- [24/05/20] LCKV initial code release. 

## Installation

You may install the dependencies with the following commands:

```sh
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

where the CUDA version is set to `12.1`. For other CUDA versions, please refer to installation instructions of [PyTorch](https://pytorch.org/get-started/locally/) and [xFormers](https://github.com/facebookresearch/xformers). See [Trouble shooting](#trouble-shooting) for more details.

## Usage

Our implementation is based on HuggingFace `transformers`. We register a new model `opt-llama` that supports the Layer-Condensed KV Cache, and a new model `cla-llama` that supports CLA. Both of them are variants of transformer `llama` models.

```python
import models # register the opt-llama and cla-llama model
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_config(config="configs/tinyllama_opt.json")
```

and now you have a randomly initialized model with the Layer-Condensed KV Cache.

### Optimization

We follows all the acceleration tricks in [tinyllama](https://github.com/jzhang38/TinyLlama), with the minimal modification to the huggingface transformers code. So we may train the model with [huggingface trainer](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) with the training speed comparable to the original tinyllama code.

To enable the optimization, add the following environment variable before running the training script:

```sh
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
```

We've done this for you in the provided training scripts. You may also refer to my [tinyllama](https://github.com/whyNLP/tinyllama) repo for a pure PyTorch implementation for the Llama model.

### Configuration

We provide some sample configuration files in the  `configs` folder. The config settings are defined in [models/configuration_llama.py](models/configuration_llama.py). You may refer to this file for more details.

#### Layer-Condensed KV Cache (LCKV)

Option 1: Modify the configurations in python:

```python
from models import OptLlamaConfig

# we have prepared a sample configuration file
config = OptLlamaConfig.from_pretrained("configs/tinyllama_opt.json")

# you may modify the configuration as you like
config.num_trained_encoders = 1      # see figure below, b-1 in the paper
config.num_encoders         = 8      # see figure below, m+b-1 in the paper
config.layer_types          = "0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_2_0" # 0: std tsfm layer; 1: layers use top layer KVs; 2: layers generate the key-value pair for other layers to use
config.target_layer         = -2     # the layer to generate the key-value pair for other layers to use
config.train_kv             = False  # add MSE loss for the key-value pair, see paper appendix

# we also supports this
config.layer_types          = "0_0_0_0_0_0_0_0_0_0_2_1_1_1_1_1_1_1_1_1_1_1" # YOCO config.
config.layer_types          = "0_0_0_0_0_0_0_0_0_0_1_1_1_1_1_1_2_1_1_1_1_1" # 2 does not necessarily have to be the last layer
```

Option 2: Modify the configurations in the shell script (via `--config_overrides`):

```sh
accelerate launch run_clm.py \
    --config_name configs/tinyllama_opt.json \
    --config_overrides model_type=opt-llama,num_encoders=8,num_trained_encoders=1,layer_types=0_1_1_1_1_1_2_0,target_layer=-2,train_kv=false \
    ...
```

Notice that some of the settings have different names and meanings compared to that in our paper. The following figure explains the correspondence:

<div align="center">
<img width="500" src="https://github.com/whyNLP/LCKV/assets/43395692/74671862-146f-492c-8d17-d0e6a7697170" />
</div>

#### Cross-Layer Attention (CLA)

Option 1: Modify the configurations in python:

```python
from models import ClaLlamaConfig

# we have prepared a sample configuration file
config = ClaLlamaConfig.from_pretrained("configs/tinyllama_cla.json")

# you may modify the configuration as you like
config.layer_types          = "2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1" # CLA-2, similar to LCKV, "1" uses the KVs from the nearest previous layer
config.layer_types          = "0_2_1_1_2_1_1_2_1_1_2_1_1_2_1_1_2_1_1_2_1_1" # CLA-3, also supports "0"
```

Option 2: Modify the configurations in the shell script (via `--config_overrides`):

```sh
accelerate launch run_clm.py \
    --config_name configs/tinyllama_cla.json \
    --config_overrides layer_types=2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1_2_1 \
    ...
```

> [!WARNING]
> The authors of CLA tuned the hyperparameters of the model architecture and training settings for the CLA model. The provided configuration files are not the optimal settings for the CLA model. You may need to change the hyperparameters for the CLA model, such as `intermediate_size`, `num_key_value_heads`, etc.

#### Grouped Layer-Condensed KV Cache (grouped LCKV)

This configuration allows you to use the KV cache from any other layer (upper or lower). The configuration is similar to LCKV, but we support multiple target layers.

Option 1: Modify the configurations in python:

```python
from models import GroupOptLlamaConfig

# we have prepared a sample configuration file
config = GroupOptLlamaConfig.from_pretrained("configs/llama_tiny_opt_group.json")

# you may modify the configuration as you like
config.num_trained_encoders = 1      # see figure below, b-1 in the paper
config.num_encoders         = 8      # see figure below, m+b-1 in the paper
config.layer_types          = "0_3_3_3_6_6_6_7" # number j in index i means the i-th layer uses the KVs from the j-th layer. Default: 0_1_2_3_4_5_6_7 is the standard transformer.

# we also supports this
config.layer_types          = "1_1_3_3_5_5_7_7" # Similar to CLA-2, but use the top layer.
config.layer_types          = "0_0_3_3_3_7_7_7" # it is fine to use lower / middle / upper layers.
```

Option 2: Modify the configurations in the shell script (via `--config_overrides`):

```sh
accelerate launch run_clm.py \
    --config_name configs/llama_tiny_opt_group.json \
    --config_overrides model_type=group-opt-llama,num_encoders=8,num_trained_encoders=1,layer_types=0_3_3_3_6_6_6_7 \
    ...
```

> [!NOTE]
> This implementation is NOT fully optimized for speed or memory.

### Training

We use the same [training script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) as the original `transformers` library. You may refer to the [official documentation](https://huggingface.co/transformers/training.html) for more details.

We provide a training script `run_clm.sh` for training a 50M parameter model on the `wikitext-103` dataset. You may run the script with:

```sh
bash run_clm.sh
```

See the script for more details. For CLA, we also provide a sample training script `run_cla.sh`. For pretraining on SlimPajama, please follow the instructions in [tinyllama-zh](https://github.com/whyNLP/tinyllama-zh) and replace the dataset with SlimPajama.

### Inference

We use the same [inference script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) as the original `transformers` library. To perform inference, you may run the following command:

```sh
bash run_generation.sh
```

You may get responses from the trained model given any prompts. See the script for more details.

### Streaming

We integrate our model with [StreamingLLM](https://github.com/mit-han-lab/streaming-llm). To perform streaming inference, you may run the following command:

```sh
bash run_streaming.sh
```

See the script for more details. The [codes](test_streaming.py) follow the [official implementation](https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py) with minimal modification.

### Evaluation

We use [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the model. You may run the following command:

```sh
python test_harness.py
```

Change the `model_args` and `tasks` in the script to evaluate different models and datasets.

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
- [xFormers](https://github.com/facebookresearch/xformers)

Please make sure to install the correct version of the packages (so long as they are consistent, the code would work). Also make sure that `nvcc` is installed and available in the path.

Our experiment environment uses `CUDA 12.1` and you may install with
```sh
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### The performance is incredibly poor

Some users have reported that the model's performance is incredibly poor and the loss does not decrease when using `torch_dtype=bfloat16` (requried by flash attention). This issue seems to be related to precision problems. Although I have not been able to reproduce this issue, a potential solution could be to use a larger learning rate. To confirm whether the issue is indeed related to precision, one could disable flash attention and use float32 instead. If the loss decreases as expected, then it is likely that the issue is related to precision.

### The code always raises exceptions

Since we start the project very early, this code base uses an old version of `transformers` (v.4.35.2). Newer versions may not be compatible with the code (I think some minor changes would fix the issue).


## Questions

> 1. Is it possible to integrate the LCKV with MQA / GQA?

Yes. The fact is that we have already done this in our experiments. Tinyllama uses 32 attention heads and 4 KV heads. We follow the same setting in our experiments. If you want to experiment with different settings, you may modify the `num_attention_heads` and `num_key_value_heads` in the configuration file.
