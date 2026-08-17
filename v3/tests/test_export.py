"""
Regression test for BUG 2: ensures merge_and_save_lora produces real model
weights on disk, not just adapter files.
"""
import os
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer

from qunart.pruner import LlamaWidthPruner
from qunart.recover import attach_lora
from qunart.exporter import merge_and_save_lora
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
    model = AutoModelForCausalLM.from_config(cfg)
    return model, cfg


class TestExport:
    def test_merge_and_save_produces_model_weights(self):
        """After prune + LoRA attach + merge_and_save_lora, the output dir
        contains model*.safetensors, config.json, and NOT only adapter files."""
        model, config = _make_tiny_llama()

        # Prune (light prune to match the config we already built)
        pruner = LlamaWidthPruner(256, 688)
        model, config = pruner.prune(model, config)

        # Attach LoRA
        target = CompressionTarget(
            lora_r=4,
            lora_alpha=8,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
        )
        model = attach_lora(model, target)

        with tempfile.TemporaryDirectory() as tmpdir:
            merge_and_save_lora(model, None, tmpdir)

            files = os.listdir(tmpdir)

            # Must contain real model weights
            has_model_safetensors = any(
                f.startswith("model") and f.endswith(".safetensors")
                for f in files
            )
            assert has_model_safetensors, f"No model*.safetensors found. Files: {files}"

            # Must contain config.json
            assert "config.json" in files, f"No config.json found. Files: {files}"

            # Must NOT contain only adapter files
            safetensors_files = [f for f in files if f.endswith(".safetensors")]
            all_adapter = all(f.startswith("adapter") for f in safetensors_files)
            assert not all_adapter, f"Only adapter files found. Files: {files}"

    def test_reload_from_pretrained(self):
        """The saved model can be reloaded with AutoModelForCausalLM.from_pretrained."""
        model, config = _make_tiny_llama()

        pruner = LlamaWidthPruner(256, 688)
        model, config = pruner.prune(model, config)

        target = CompressionTarget(
            lora_r=4,
            lora_alpha=8,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
        )
        model = attach_lora(model, target)

        with tempfile.TemporaryDirectory() as tmpdir:
            merge_and_save_lora(model, None, tmpdir)

            # Reload
            reloaded = AutoModelForCausalLM.from_pretrained(tmpdir)
            assert reloaded is not None

            # Verify forward pass works
            reloaded.eval()
            input_ids = torch.randint(0, 1000, (1, 4))
            with torch.no_grad():
                out = reloaded(input_ids)
            assert out.logits.shape == (1, 4, 1000)
