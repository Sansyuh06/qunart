from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from .qubo import QUBOSolver
from .importance import compute_mlp_neuron_importance


class BasePruner(ABC):
    @abstractmethod
    def prune(self, model, config) -> Tuple[torch.nn.Module, object]:
        """Return the pruned model and updated config."""
        pass


class LlamaWidthPruner(BasePruner):
    """
    Structured width pruning for Llama/Qwen2/Mistral-style models.

    Keeps head_dim constant so rotary embeddings do not need rebuilding.
    Reduces hidden_size, intermediate_size, and the number of attention heads
    proportionally, then slices all weights in place and updates the config.
    """

    SUPPORTED_ARCHS = (
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "MistralForCausalLM",
    )

    def __init__(
        self,
        new_hidden_size: int,
        new_intermediate_size: int,
        qubo_solver: Optional[QUBOSolver] = None,
    ):
        self.new_h = new_hidden_size
        self.new_i = new_intermediate_size
        self.qubo = qubo_solver or QUBOSolver("greedy")

    def prune(self, model, config):
        old_h = config.hidden_size
        old_i = config.intermediate_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = old_h // num_heads

        if self.new_h % head_dim != 0:
            raise ValueError(
                f"new_hidden_size ({self.new_h}) must be divisible by head_dim ({head_dim})"
            )

        new_num_heads = self.new_h // head_dim
        new_num_kv_heads = max(1, round(num_kv_heads * self.new_h / old_h))
        # preserve the original GQA grouping ratio if exact divisibility fails
        if new_num_heads % new_num_kv_heads != 0:
            new_num_kv_heads = max(1, round(new_num_heads / (num_heads / num_kv_heads)))
        new_kv_dim = new_num_kv_heads * head_dim

        # --- embeddings ---
        model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, : self.new_h]
        if model.model.embed_tokens.bias is not None:
            model.model.embed_tokens.bias.data = model.model.embed_tokens.bias.data[: self.new_h]

        # --- lm head (if not tied) ---
        if hasattr(model, "lm_head") and not config.tie_word_embeddings:
            model.lm_head.weight.data = model.lm_head.weight.data[:, : self.new_h]
            if model.lm_head.bias is not None:
                model.lm_head.bias.data = model.lm_head.bias.data[: self.new_h]

        # --- layers ---
        for layer in model.model.layers:
            # norms
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, norm_name)
                norm.weight.data = norm.weight.data[: self.new_h]
                if getattr(norm, "bias", None) is not None:
                    norm.bias.data = norm.bias.data[: self.new_h]

            # attention: slice weights and update module bookkeeping
            attn = layer.self_attn
            attn.q_proj.weight.data = attn.q_proj.weight.data[: self.new_h, : self.new_h]
            attn.k_proj.weight.data = attn.k_proj.weight.data[:new_kv_dim, : self.new_h]
            attn.v_proj.weight.data = attn.v_proj.weight.data[:new_kv_dim, : self.new_h]
            attn.o_proj.weight.data = attn.o_proj.weight.data[: self.new_h, : self.new_h]

            for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                proj.out_features = proj.weight.size(0)
                proj.in_features = proj.weight.size(1)
                if proj.bias is not None:
                    proj.bias.data = proj.bias.data[: proj.out_features]

            attn.num_heads = new_num_heads
            attn.num_key_value_heads = new_num_kv_heads
            attn.num_key_value_groups = new_num_heads // new_num_kv_heads
            attn.hidden_size = self.new_h
            # head_dim unchanged; rotary_emb stays valid

            # mlp: use QUBO to select which neurons survive, then slice
            mlp = layer.mlp
            self._prune_mlp(mlp)

        # --- update config ---
        config.hidden_size = self.new_h
        config.intermediate_size = self.new_i
        config.num_attention_heads = new_num_heads
        config.num_key_value_heads = new_num_kv_heads

        return model, config

    def _prune_mlp(self, mlp):
        old_i = mlp.gate_proj.weight.size(0)
        target_i = min(self.new_i, old_i)

        importances = compute_mlp_neuron_importance(
            mlp.gate_proj.weight.data,
            mlp.up_proj.weight.data,
            mlp.down_proj.weight.data,
        )
        keep = self.qubo.solve(importances, target_i)
        keep = torch.tensor(keep, device=mlp.gate_proj.weight.device, dtype=torch.long)

        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[keep]
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[keep]
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, keep]

        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            proj.out_features = proj.weight.size(0)
            proj.in_features = proj.weight.size(1)
            if proj.bias is not None:
                proj.bias.data = proj.bias.data[: proj.out_features]


class Phi3WidthPruner(BasePruner):
    """
    Structured width pruning for Microsoft Phi-3 models.

    Phi-3 uses fused projections:
      - qkv_proj  (q + k + v fused)
      - gate_up_proj (gate + up fused)

    Like LlamaWidthPruner, this keeps head_dim constant and slices the fused
    weights in place, then re-assembles them.
    """

    SUPPORTED_ARCHS = ("Phi3ForCausalLM",)

    def __init__(
        self,
        new_hidden_size: int,
        new_intermediate_size: int,
        qubo_solver: Optional[QUBOSolver] = None,
    ):
        self.new_h = new_hidden_size
        self.new_i = new_intermediate_size
        self.qubo = qubo_solver or QUBOSolver("greedy")

    def prune(self, model, config):
        old_h = config.hidden_size
        old_i = config.intermediate_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = old_h // num_heads

        if self.new_h % head_dim != 0:
            raise ValueError(
                f"new_hidden_size ({self.new_h}) must be divisible by head_dim ({head_dim})"
            )

        new_num_heads = self.new_h // head_dim
        new_num_kv_heads = max(1, round(num_kv_heads * self.new_h / old_h))
        if new_num_heads % new_num_kv_heads != 0:
            new_num_kv_heads = max(1, round(new_num_heads / (num_heads / num_kv_heads)))

        new_q_dim = new_num_heads * head_dim
        new_kv_dim = new_num_kv_heads * head_dim

        old_q_dim = num_heads * head_dim
        old_kv_dim = num_kv_heads * head_dim

        # --- embeddings ---
        model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, : self.new_h]
        if model.model.embed_tokens.bias is not None:
            model.model.embed_tokens.bias.data = model.model.embed_tokens.bias.data[: self.new_h]

        # --- lm head (if not tied) ---
        if hasattr(model, "lm_head") and not config.tie_word_embeddings:
            model.lm_head.weight.data = model.lm_head.weight.data[:, : self.new_h]
            if model.lm_head.bias is not None:
                model.lm_head.bias.data = model.lm_head.bias.data[: self.new_h]

        # --- layers ---
        for layer in model.model.layers:
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, norm_name)
                norm.weight.data = norm.weight.data[: self.new_h]
                if getattr(norm, "bias", None) is not None:
                    norm.bias.data = norm.bias.data[: self.new_h]

            attn = layer.self_attn
            qkv = attn.qkv_proj.weight.data

            # qkv is stacked vertically: [q | k | v]
            q_part = qkv[:old_q_dim, : self.new_h]
            k_part = qkv[old_q_dim : old_q_dim + old_kv_dim, : self.new_h]
            v_part = qkv[old_q_dim + old_kv_dim : old_q_dim + 2 * old_kv_dim, : self.new_h]

            new_qkv = torch.cat(
                [q_part[:new_q_dim], k_part[:new_kv_dim], v_part[:new_kv_dim]], dim=0
            )
            attn.qkv_proj.weight.data = new_qkv
            attn.qkv_proj.out_features = new_qkv.size(0)
            attn.qkv_proj.in_features = self.new_h
            if attn.qkv_proj.bias is not None:
                attn.qkv_proj.bias.data = attn.qkv_proj.bias.data[: new_qkv.size(0)]

            attn.o_proj.weight.data = attn.o_proj.weight.data[: self.new_h, :new_q_dim]
            attn.o_proj.out_features = self.new_h
            attn.o_proj.in_features = new_q_dim
            if attn.o_proj.bias is not None:
                attn.o_proj.bias.data = attn.o_proj.bias.data[: self.new_h]

            attn.num_heads = new_num_heads
            attn.num_key_value_heads = new_num_kv_heads
            attn.num_key_value_groups = new_num_heads // new_num_kv_heads
            attn.hidden_size = self.new_h

            # mlp
            mlp = layer.mlp
            self._prune_phi3_mlp(mlp, old_i)

        # --- update config ---
        config.hidden_size = self.new_h
        config.intermediate_size = self.new_i
        config.num_attention_heads = new_num_heads
        config.num_key_value_heads = new_num_kv_heads

        return model, config

    def _prune_phi3_mlp(self, mlp, old_i: int):
        gate_up = mlp.gate_up_proj.weight.data
        down = mlp.down_proj.weight.data

        # gate_up is stacked vertically: [gate | up]
        gate = gate_up[:old_i]
        up = gate_up[old_i:]

        target_i = min(self.new_i, old_i)

        importances = (
            gate.norm(dim=1).cpu().numpy()
            + up.norm(dim=1).cpu().numpy()
            + down.norm(dim=0).cpu().numpy()
        )
        keep = self.qubo.solve(importances, target_i)
        keep = torch.tensor(keep, device=gate_up.device, dtype=torch.long)

        new_gate = gate[keep]
        new_up = up[keep]
        new_down = down[:, keep]

        mlp.gate_up_proj.weight.data = torch.cat([new_gate, new_up], dim=0)
        mlp.gate_up_proj.out_features = 2 * target_i
        mlp.gate_up_proj.in_features = self.new_h
        if mlp.gate_up_proj.bias is not None:
            mlp.gate_up_proj.bias.data = mlp.gate_up_proj.bias.data[: 2 * target_i]

        mlp.down_proj.weight.data = new_down
        mlp.down_proj.out_features = self.new_h
        mlp.down_proj.in_features = target_i
        if mlp.down_proj.bias is not None:
            mlp.down_proj.bias.data = mlp.down_proj.bias.data[: self.new_h]
