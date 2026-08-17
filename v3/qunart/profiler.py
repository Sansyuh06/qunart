from typing import Dict, Any


def profile_model(model, config) -> Dict[str, Any]:
    """Return a parameter and architecture summary."""
    total = sum(p.numel() for p in model.parameters())
    hidden = getattr(config, "hidden_size", 0)
    heads = getattr(config, "num_attention_heads", 0)
    kv_heads = getattr(config, "num_key_value_heads", heads)
    intermediate = getattr(config, "intermediate_size", 0)
    layers = getattr(config, "num_hidden_layers", 0)
    vocab = getattr(config, "vocab_size", 0)
    archs = getattr(config, "architectures", None)
    arch = archs[0] if archs else getattr(model, "__class__", type(model)).__name__

    return {
        "total_params": total,
        "hidden_size": hidden,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": hidden // heads if heads else 0,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "architecture": arch,
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", True),
    }


def total_params_at(profile: Dict[str, Any], new_h: int, new_i: int) -> int:
    """Approximate total parameters for a Llama-style width-pruned model."""
    vocab = profile["vocab_size"]
    layers = profile["num_layers"]
    heads = profile["num_attention_heads"]
    kv_heads = profile["num_key_value_heads"]
    head_dim = profile["head_dim"]
    tie = profile["tie_word_embeddings"]

    new_num_heads = new_h // head_dim
    target_kv = max(1, round(kv_heads * new_h / profile["hidden_size"]))
    new_num_kv_heads = 1
    for d in range(min(new_num_heads, target_kv), 0, -1):
        if new_num_heads % d == 0:
            new_num_kv_heads = d
            break
    new_kv_dim = new_num_kv_heads * head_dim

    embedding = vocab * new_h
    lm_head = 0 if tie else vocab * new_h

    # q_proj + o_proj: full hidden -> hidden, k/v: kv_dim -> hidden
    attn_per_layer = (new_h * new_h) + 2 * (new_kv_dim * new_h) + (new_h * new_h)
    mlp_per_layer = 2 * (new_i * new_h) + (new_h * new_i)
    norms_per_layer = 2 * new_h

    final_norm = new_h
    total = embedding + lm_head + final_norm + layers * (attn_per_layer + mlp_per_layer + norms_per_layer)
    return int(total)