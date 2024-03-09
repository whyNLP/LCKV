# ALGPT

A variation of ALBERT that applies to GPT-2, with few parameters, fast inference and many interesting features. Working in progress.

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