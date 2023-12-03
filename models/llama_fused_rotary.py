from flash_attn.layers.rotary import apply_rotary_emb

def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    assert unsqueeze_dim == 1, "fused rotary pos emb only supports unsqueeze_dim=1"
    assert q.shape[-1] == cos.shape[-1], "q and cos must have the same embedding dimension"
    if len(position_ids.shape) == 2:
        assert position_ids.shape[0] == 1, "position_ids must have a batch dimension of 1"
        position_ids = position_ids[0]
    
    ro_dim = cos.shape[-1]
    cos = cos[position_ids][..., :ro_dim // 2]
    sin = sin[position_ids][..., :ro_dim // 2]
    q_embed = apply_rotary_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
    k_embed = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
    return q_embed, k_embed