"""
Regression test for BUG 4: ensures LoRA adapters are injected into
the correct modules for each architecture.
"""
import pytest
from transformers import AutoModelForCausalLM, LlamaConfig

from qunart.pruner import LlamaWidthPruner, Phi3WidthPruner
from qunart.recover import attach_lora
from qunart.config import CompressionTarget


def _make_tiny_llama():
    cfg = LlamaConfig(
        hidden_size=256,
        intermediate_size=688,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1000,
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    return AutoModelForCausalLM.from_config(cfg), cfg


def _make_tiny_phi3():
    try:
        from transformers import Phi3Config
    except ImportError:
        pytest.skip("Phi3Config not available")
    cfg = Phi3Config(
        hidden_size=256,
        intermediate_size=688,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1000,
        tie_word_embeddings=False,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    return AutoModelForCausalLM.from_config(cfg), cfg


class TestLoraTargetsLlama:
    def test_llama_lora_injection_count(self):
        """For Llama, all 7 target modules per layer should get LoRA."""
        model, config = _make_tiny_llama()
        pruner = LlamaWidthPruner(256, 688)
        model, config = pruner.prune(model, config)

        llama_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        target = CompressionTarget(
            lora_r=4, lora_alpha=8,
            lora_target_modules=llama_targets,
        )
        peft_model = attach_lora(model, target)

        # Count lora_A modules (one per injected target)
        lora_a_count = sum(
            1 for n, _ in peft_model.named_modules()
            if "lora_A" in n and "default" in n
        )
        expected = config.num_hidden_layers * len(llama_targets)
        assert lora_a_count == expected, \
            f"Expected {expected} lora_A modules, got {lora_a_count}"


class TestLoraTargetsPhi3:
    def test_phi3_lora_injection_count(self):
        """For Phi-3, all 4 target modules per layer should get LoRA."""
        model, config = _make_tiny_phi3()
        pruner = Phi3WidthPruner(256, 688)
        model, config = pruner.prune(model, config)

        phi3_targets = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        target = CompressionTarget(
            lora_r=4, lora_alpha=8,
            lora_target_modules=phi3_targets,
        )
        peft_model = attach_lora(model, target)

        lora_a_count = sum(
            1 for n, _ in peft_model.named_modules()
            if "lora_A" in n and "default" in n
        )
        expected = config.num_hidden_layers * len(phi3_targets)  # 2 * 4 = 8
        assert lora_a_count == expected, \
            f"Expected {expected} lora_A modules, got {lora_a_count}"

    def test_phi3_default_targets_are_wrong(self):
        """If we use default Llama targets on Phi-3, we get fewer adapters.
        This is the BUG 4 regression test."""
        model, config = _make_tiny_phi3()
        pruner = Phi3WidthPruner(256, 688)
        model, config = pruner.prune(model, config)

        # Default (wrong) targets for Phi-3
        wrong_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        target = CompressionTarget(
            lora_r=4, lora_alpha=8,
            lora_target_modules=wrong_targets,
        )
        peft_model = attach_lora(model, target)

        lora_a_count = sum(
            1 for n, _ in peft_model.named_modules()
            if "lora_A" in n and "default" in n
        )
        # With wrong targets, fewer modules get LoRA
        correct_count = config.num_hidden_layers * 4  # 8
        assert lora_a_count < correct_count, \
            "Wrong targets should inject fewer adapters than correct targets"
