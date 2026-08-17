"""
Regression gate: ensures pruned models have correct shapes and can forward pass.
Must run on CPU, no network, in seconds.
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from qunart.pruner import LlamaWidthPruner, Phi3WidthPruner
from qunart.profiler import profile_model, total_params_at
from qunart.qubo import QUBOSolver


# ---- Helpers ----

def _make_llama(
    hidden_size=512,
    intermediate_size=1376,
    num_attention_heads=8,
    num_key_value_heads=2,
    num_hidden_layers=2,
    vocab_size=1000,
):
    cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        tie_word_embeddings=False,
        use_cache=True,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model, cfg


def _make_phi3(
    hidden_size=512,
    intermediate_size=1376,
    num_attention_heads=8,
    num_key_value_heads=2,
    num_hidden_layers=2,
    vocab_size=1000,
):
    try:
        from transformers import Phi3Config
    except ImportError:
        pytest.skip("Phi3Config not available in this transformers version")

    cfg = Phi3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        tie_word_embeddings=False,
        use_cache=True,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model, cfg


# ---- Llama Tests ----

class TestLlamaPruner:
    """Tests for LlamaWidthPruner (covers Llama/Qwen2/Mistral)."""

    def setup_method(self):
        self.old_h = 512
        self.old_i = 1376
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = self.old_h // self.num_heads  # 64
        self.new_h = 256  # 50% width prune
        self.new_i = 688  # ~50% intermediate
        self.model, self.config = _make_llama(
            hidden_size=self.old_h,
            intermediate_size=self.old_i,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
        )

    def test_shapes_after_prune(self):
        """Every weight matrix has the expected shape after a 50% width prune."""
        pruner = LlamaWidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)

        assert config.hidden_size == self.new_h
        assert config.intermediate_size == self.new_i
        new_num_heads = self.new_h // self.head_dim  # 4
        new_kv_heads = config.num_key_value_heads
        new_kv_dim = new_kv_heads * self.head_dim

        # Embedding
        assert model.model.embed_tokens.weight.shape == (1000, self.new_h)

        # lm_head (not tied)
        assert model.lm_head.weight.shape == (1000, self.new_h)

        for layer in model.model.layers:
            attn = layer.self_attn
            # q_proj: (new_h, new_h)
            assert attn.q_proj.weight.shape == (self.new_h, self.new_h)
            # k_proj: (new_kv_dim, new_h)
            assert attn.k_proj.weight.shape == (new_kv_dim, self.new_h)
            # v_proj: (new_kv_dim, new_h)
            assert attn.v_proj.weight.shape == (new_kv_dim, self.new_h)
            # o_proj: (new_h, new_h)
            assert attn.o_proj.weight.shape == (self.new_h, self.new_h)

            mlp = layer.mlp
            # gate_proj: (new_i, new_h)
            assert mlp.gate_proj.weight.shape == (self.new_i, self.new_h), \
                f"gate_proj: expected ({self.new_i}, {self.new_h}), got {mlp.gate_proj.weight.shape}"
            # up_proj: (new_i, new_h)
            assert mlp.up_proj.weight.shape == (self.new_i, self.new_h), \
                f"up_proj: expected ({self.new_i}, {self.new_h}), got {mlp.up_proj.weight.shape}"
            # down_proj: (new_h, new_i)
            assert mlp.down_proj.weight.shape == (self.new_h, self.new_i), \
                f"down_proj: expected ({self.new_h}, {self.new_i}), got {mlp.down_proj.weight.shape}"

    def test_forward_pass(self):
        """A real forward pass returns logits with correct shape."""
        pruner = LlamaWidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (2, 16, 1000)

    def test_generate(self):
        """model.generate(max_new_tokens=8) runs without raising."""
        pruner = LlamaWidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=8, do_sample=False)
        assert out.shape[1] == 4 + 8

    def test_param_count_matches_estimate(self):
        """Realised param count is within 2% of total_params_at()."""
        pruner = LlamaWidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        profile = profile_model(model, config)
        estimated = total_params_at(profile, self.new_h, self.new_i)
        actual = sum(p.numel() for p in model.parameters())
        pct_diff = abs(actual - estimated) / max(estimated, 1) * 100
        assert pct_diff <= 2.0, f"Param mismatch: actual={actual}, estimated={estimated}, diff={pct_diff:.1f}%"

    def test_head_dim_constant(self):
        """head_dim is unchanged and GQA grouping is valid."""
        pruner = LlamaWidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        new_num_heads = config.num_attention_heads
        new_kv_heads = config.num_key_value_heads
        new_head_dim = config.hidden_size // new_num_heads
        assert new_head_dim == self.head_dim
        assert new_num_heads % new_kv_heads == 0

    def test_no_attributeerror_on_embedding_bias(self):
        """Pruning a model with nn.Embedding does not raise AttributeError (BUG 1)."""
        # Just ensure prune completes without AttributeError
        pruner = LlamaWidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        assert config.hidden_size == self.new_h


# ---- Phi-3 Tests ----

class TestPhi3Pruner:
    """Tests for Phi3WidthPruner (fused projections)."""

    def setup_method(self):
        self.old_h = 512
        self.old_i = 1376
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = self.old_h // self.num_heads  # 64
        self.new_h = 256
        self.new_i = 688
        try:
            self.model, self.config = _make_phi3(
                hidden_size=self.old_h,
                intermediate_size=self.old_i,
                num_attention_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
            )
        except Exception:
            pytest.skip("Phi3Config not available")

    def test_shapes_after_prune(self):
        pruner = Phi3WidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)

        assert config.hidden_size == self.new_h
        assert config.intermediate_size == self.new_i
        new_num_heads = self.new_h // self.head_dim
        new_kv_heads = config.num_key_value_heads
        new_q_dim = new_num_heads * self.head_dim
        new_kv_dim = new_kv_heads * self.head_dim
        expected_qkv = new_q_dim + 2 * new_kv_dim

        for layer in model.model.layers:
            attn = layer.self_attn
            # qkv_proj: (q+k+v, new_h)
            assert attn.qkv_proj.weight.shape == (expected_qkv, self.new_h)
            # o_proj: (new_h, new_q_dim)
            assert attn.o_proj.weight.shape == (self.new_h, new_q_dim)

            mlp = layer.mlp
            # gate_up_proj: (2*new_i, new_h)
            assert mlp.gate_up_proj.weight.shape == (2 * self.new_i, self.new_h), \
                f"gate_up_proj: expected ({2*self.new_i}, {self.new_h}), got {mlp.gate_up_proj.weight.shape}"
            # down_proj: (new_h, new_i)
            assert mlp.down_proj.weight.shape == (self.new_h, self.new_i), \
                f"down_proj: expected ({self.new_h}, {self.new_i}), got {mlp.down_proj.weight.shape}"

    def test_forward_pass(self):
        pruner = Phi3WidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (2, 16, 1000)

    def test_generate(self):
        pruner = Phi3WidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=8, do_sample=False)
        assert out.shape[1] == 4 + 8

    def test_param_count_matches_estimate(self):
        pruner = Phi3WidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        profile = profile_model(model, config)
        estimated = total_params_at(profile, self.new_h, self.new_i)
        actual = sum(p.numel() for p in model.parameters())
        pct_diff = abs(actual - estimated) / max(estimated, 1) * 100
        assert pct_diff <= 2.0, f"Param mismatch: actual={actual}, estimated={estimated}, diff={pct_diff:.1f}%"

    def test_head_dim_constant(self):
        pruner = Phi3WidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        new_num_heads = config.num_attention_heads
        new_kv_heads = config.num_key_value_heads
        new_head_dim = config.hidden_size // new_num_heads
        assert new_head_dim == self.head_dim
        assert new_num_heads % new_kv_heads == 0

    def test_no_attributeerror_on_embedding_bias(self):
        pruner = Phi3WidthPruner(self.new_h, self.new_i)
        model, config = pruner.prune(self.model, self.config)
        assert config.hidden_size == self.new_h
