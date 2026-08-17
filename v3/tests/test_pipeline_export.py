"""
End-to-end regression test for BUG 2 at the pipeline orchestration level.
Ensures CompressionPipeline.run() writes full merged base weights (model*.safetensors),
not merely adapter files, and that the result reloads via AutoModelForCausalLM.from_pretrained.
"""
import os
import tempfile
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from qunart import CompressionPipeline, CompressionTarget
import qunart.pipeline


def _create_dummy_llama(save_dir: str):
    """Create and save a tiny in-memory Llama model and tokenizer for offline testing."""
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=688,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1000,
        tie_word_embeddings=False,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(save_dir)
    return save_dir


def test_pipeline_run_merges_and_exports_base_weights(monkeypatch):
    """
    Verify that CompressionPipeline.run() saves real merged base weights
    (model*.safetensors), and not just adapter weights.
    """
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=688,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1000,
        tie_word_embeddings=False,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.architectures = ["LlamaForCausalLM"]
    model = AutoModelForCausalLM.from_config(config)

    # Monkeypatch loader.load to return our in-memory model
    monkeypatch.setattr(
        qunart.loader.ModelLoader,
        "load",
        lambda self, model_path: (model, None, config),
    )

    # Monkeypatch finetune to be a fast offline no-op
    monkeypatch.setattr(
        qunart.pipeline,
        "finetune",
        lambda model, tokenizer, target, dataset_name=None: model,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        target = CompressionTarget(
            target_params=1500000,  # ~50% width reduction
            finetune_steps=1,
            lora_r=4,
            lora_alpha=8,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            torch_dtype="float32",
            device="cpu",
        )

        pipeline = CompressionPipeline(target)
        pipeline.run("dummy_path", output_dir)

        saved_files = os.listdir(output_dir)

        # Must contain real model weights
        has_model = any(
            f.startswith("model") and f.endswith(".safetensors") for f in saved_files
        )
        assert has_model, f"Pipeline failed to save model*.safetensors. Files in output: {saved_files}"

        # Must NOT contain only adapter weights
        has_only_adapter = all(
            f.startswith("adapter") for f in saved_files if f.endswith(".safetensors")
        )
        assert not has_only_adapter, f"Output contains only adapter files! Files: {saved_files}"

        # Must reload and complete a forward pass
        reloaded = AutoModelForCausalLM.from_pretrained(output_dir)
        reloaded.eval()
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            out = reloaded(input_ids)
        assert out.logits.shape == (1, 4, 1000)

