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

```sh
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
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
- PyTorch
- Flash-Attn
- XFormers

Please make sure to install the correct version of the packages (so long as they are consistent, the code would work). Also make sure that `nvcc` is installed and available in the path.

Our environment is set to `CUDA 12.1` and you may install with
```sh
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


