import torch.nn as nn
from xformers.ops.swiglu_op import swiglu

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        assert config.hidden_act in ('silu', 'swish'), f"Fused SwiGLU requires silu / swish as activate function, but {config.hidden_act} found."
        assert config.pretraining_tp == 1, "Fused SwiGLU requires pretraining_tp == 1"

    def forward(self, x):
        return swiglu(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.up_proj.weight,
            self.up_proj.bias,
            self.down_proj.weight,
            self.down_proj.bias,
            op=None
        )
