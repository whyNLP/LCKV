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

The code base for project **Layer-Condensed KV Cache**, a new variant of transformer decoders in which queries of all layers are paired with keys and values of just the top layer. It reduces the memory and computation cost, reduces the number of parameters, significantly improves the inference throughput with comparable or better task performance. The paper "[Layer-Condensed KV Cache for Efficient Inference of Large Language Models](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/)" is submitted to ACL 2024.

This work is inspired by [Probabilistic Transformer](https://github.com/whyNLP/Probabilistic-Transformer), where we consider the stacking layer structure of a transformer as an iterative process of improving token representation.

## Installation

You may install the dependencies with the following commands:

```sh
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

where the CUDA version is set to `12.1`. For other CUDA versions, please refer to installation instructions of [PyTorch](https://pytorch.org/get-started/locally/) and [xFormers](https://github.com/facebookresearch/xformers). See [Trouble shooting](#trouble-shooting) for more details.

## Usage

Our implementation is based on HuggingFace `transformers` where we register a new model `opt-llama` that supports the Layer-Condensed KV Cache.

```python
import models # register the opt-llama model
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

Notice that some of the settings have different names and meanings compared to that in our paper. The following figure explains the correspondence:

<div align="center">
<img width="500" src="https://github.com/whyNLP/LCKV/assets/43395692/74671862-146f-492c-8d17-d0e6a7697170" />
</div>

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

See the script for more details.

### Streaming

We integrate our model with [StreamingLLM](https://github.com/mit-han-lab/streaming-llm). To perform streaming inference, you may run the following command:

```sh
bash run_streaming.sh
```

See the script for more details. The [codes](test_streaming.py) follow the [official implementation](https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py) with minimal modification.

> [!WARNING]
> The script `run_streaming.py` is not supported yet.

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

