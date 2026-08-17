"""Smoke test: prune TinyLlama → LoRA recover → save → reload → generate."""
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qunart.config import CompressionTarget
from qunart.loader import ModelLoader
from qunart.profiler import profile_model, total_params_at
from qunart.pruner import LlamaWidthPruner
from qunart.recover import attach_lora, finetune
from qunart.exporter import merge_and_save_lora
from qunart.pipeline import _lora_targets_for_arch


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "results/smoke_model"
    results_file = "results/smoke.txt"
    os.makedirs("results", exist_ok=True)

    output_lines = []
    def log(msg):
        print(msg)
        output_lines.append(msg)

    log("=" * 60)
    log("qunart Smoke Test")
    log("=" * 60)

    # Load
    log("\n[1/6] Loading TinyLlama...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    config = model.config
    profile = profile_model(model, config)
    orig_params = profile["total_params"]
    log(f"   Original params: {orig_params:,}")

    # Generate before pruning
    log("\n[2/6] Generating (before pruning)...")
    prompt = "The key advantage of on-device AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    before_text = tokenizer.decode(out[0], skip_special_tokens=True)
    log(f"   Before: {before_text}")

    # Plan: ~60% params
    log("\n[3/6] Planning (~60% params)...")
    target_params = int(orig_params * 0.6)
    head_dim = profile["head_dim"]
    old_h = profile["hidden_size"]
    old_i = profile["intermediate_size"]

    # Binary search
    lo, hi = 0.1, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        h = max(head_dim, round(old_h * mid / head_dim) * head_dim)
        i = max(1, round(old_i * mid))
        if total_params_at(profile, h, i) > target_params:
            hi = mid
        else:
            lo = mid
    new_h = max(head_dim, round(old_h * lo / head_dim) * head_dim)
    new_i = max(1, round(old_i * lo))
    estimated = total_params_at(profile, new_h, new_i)
    log(f"   Target: {target_params:,}, new_h={new_h}, new_i={new_i}")
    log(f"   Estimated: {estimated:,}")

    # Prune
    log("\n[4/6] Pruning...")
    pruner = LlamaWidthPruner(new_h, new_i)
    model, config = pruner.prune(model, config)
    actual = sum(p.numel() for p in model.parameters())
    log(f"   Actual params after prune: {actual:,}")
    log(f"   Compression: {orig_params/actual:.2f}x")

    # LoRA recover (just 50 steps for smoke test)
    log("\n[5/6] LoRA recovery (50 steps)...")
    target = CompressionTarget(
        finetune_steps=50,
        finetune_lr=2e-4,
        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        max_seq_length=256,
        recovery_samples=500,
        torch_dtype="float16" if device == "cuda" else "float32",
        device=device,
        lora_target_modules=_lora_targets_for_arch(profile["architecture"]),
    )
    model = attach_lora(model, target)
    model = finetune(model, tokenizer, target)

    # Save with merge
    log("\n[6/6] Merging LoRA and saving...")
    merge_and_save_lora(model, tokenizer, output_dir)

    # Verify files
    saved_files = os.listdir(output_dir)
    has_model = any(f.startswith("model") and f.endswith(".safetensors") for f in saved_files)
    log(f"   Saved files: {saved_files}")
    log(f"   Contains model weights: {has_model}")

    # Disk size
    total_bytes = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
    log(f"   Disk size: {total_bytes / 1e6:.1f} MB")

    # Reload and generate
    log("\nReloading from disk...")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        output_dir, torch_dtype=dtype, device_map=device
    )
    reloaded.eval()
    reload_params = sum(p.numel() for p in reloaded.parameters())
    log(f"   Reloaded params: {reload_params:,}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = reloaded.generate(**inputs, max_new_tokens=50, do_sample=False)
    after_text = tokenizer.decode(out[0], skip_special_tokens=True)
    log(f"   After: {after_text}")

    # Summary
    log("\n" + "=" * 60)
    log("SMOKE TEST SUMMARY")
    log("=" * 60)
    log(f"Original params:  {orig_params:,}")
    log(f"Pruned params:    {actual:,} ({actual/orig_params*100:.1f}%)")
    log(f"Disk size:        {total_bytes / 1e6:.1f} MB")
    log(f"Model reloads:    YES")
    log(f"Forward pass:     YES")
    log(f"Before: {before_text[:100]}...")
    log(f"After:  {after_text[:100]}...")

    # Write results
    with open(results_file, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
