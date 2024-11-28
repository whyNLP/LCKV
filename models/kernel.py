from liger_kernel.transformers.monkey_patch import (
    LigerCrossEntropyLoss,
    LigerRMSNorm,
    LigerSwiGLUMLP,
    PreTrainedModel,
    _bind_method_to_module,
    _patch_rms_norm_module,
    llama_lce_forward,
)

from .ops_rope import SingleLigerRopeFunction


def liger_rotary(q, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim).
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """

    return SingleLigerRopeFunction.apply(q, cos, sin, None, unsqueeze_dim)


def apply_liger_kernel_to_lckv_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in LCKV models.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.llama import modeling_llama

    from . import modeling_lckv

    if rope:
        modeling_lckv.apply_rotary = liger_rotary
    if rms_norm:
        modeling_llama.LlamaRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_llama.LlamaMLP = LigerSwiGLUMLP
    if cross_entropy:
        modeling_llama.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_lckv.LlamaForCausalLM.forward = llama_lce_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)

        base_model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(
                    decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward
                )
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
