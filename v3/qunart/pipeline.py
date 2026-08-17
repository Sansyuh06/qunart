import os
from typing import Optional, Tuple

from .config import CompressionTarget
from .loader import ModelLoader
from .profiler import profile_model, total_params_at
from .pruner import LlamaWidthPruner, Phi3WidthPruner
from .recover import attach_lora, finetune
from .exporter import merge_and_save_lora, export_gguf, export_onnx
from .qubo import QUBOSolver


# Architecture-specific LoRA target modules
_LORA_TARGETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


SUPPORTED = (
    LlamaWidthPruner.SUPPORTED_ARCHS
    + Phi3WidthPruner.SUPPORTED_ARCHS
)


def _lora_targets_for_arch(arch: str) -> list:
    """Return the correct LoRA target module names for the architecture."""
    if arch in Phi3WidthPruner.SUPPORTED_ARCHS:
        return _LORA_TARGETS["phi3"]
    return _LORA_TARGETS["llama"]


class CompressionPipeline:
    """End-to-end: load → profile → plan → prune → recover → export."""

    def __init__(self, target: CompressionTarget):
        self.target = target
        self.loader = ModelLoader(
            device=target.device, torch_dtype=target.torch_dtype
        )

    def run(
        self,
        model_path: str,
        output_dir: str,
        dataset_name: Optional[str] = None,
        export_format: str = "hf",
    ):
        print("=" * 60)
        print("Qunart — Universal LLM Compression")
        print("=" * 60)

        print("\n[1/5] Loading model...")
        model, tokenizer, config = self.loader.load(model_path)
        profile = profile_model(model, config)
        print(f"      Architecture : {profile['architecture']}")
        print(f"      Original params: {profile['total_params']:,}")
        print(f"      Hidden/Heads/I : {profile['hidden_size']} / {profile['num_attention_heads']} / {profile['intermediate_size']}")

        print("\n[2/5] Planning compression...")
        target_params = self._resolve_target_params(profile)
        new_h, new_i = self._plan_sizes(profile, target_params)
        print(f"      Target params: {target_params:,}")
        print(f"      New hidden_size: {new_h}, intermediate_size: {new_i}")
        estimated = total_params_at(profile, new_h, new_i)
        print(f"      Estimated final params: {estimated:,}")

        print("\n[3/5] Pruning...")
        arch = profile["architecture"]
        qubo_solver = QUBOSolver(
            method=self.target.selection_method,
            lambda_redundancy=self.target.lambda_redundancy,
        )
        if arch in LlamaWidthPruner.SUPPORTED_ARCHS:
            pruner = LlamaWidthPruner(new_h, new_i, qubo_solver=qubo_solver)
        elif arch in Phi3WidthPruner.SUPPORTED_ARCHS:
            pruner = Phi3WidthPruner(new_h, new_i, qubo_solver=qubo_solver)
        else:
            raise ValueError(
                f"Architecture '{arch}' is not yet supported. "
                f"Supported: {SUPPORTED}"
            )
        model, config = pruner.prune(model, config)

        # BUG 6 fix: verify estimated vs actual param count
        actual_params = sum(p.numel() for p in model.parameters())
        print(f"      Actual params after prune: {actual_params:,}")
        print(f"      Estimated params:          {estimated:,}")
        pct_diff = abs(actual_params - estimated) / max(estimated, 1) * 100
        if pct_diff > 2.0:
            raise ValueError(
                f"Parameter count mismatch: actual={actual_params:,}, "
                f"estimated={estimated:,} ({pct_diff:.1f}% difference). "
                f"The profiler estimator may need updating."
            )

        print("\n[4/5] Recovery fine-tuning (LoRA)...")
        # BUG 4 fix: set architecture-appropriate LoRA targets
        if self.target.lora_target_modules == CompressionTarget().lora_target_modules:
            self.target.lora_target_modules = _lora_targets_for_arch(arch)
        model = attach_lora(model, self.target)
        # Verify LoRA injection count
        expected_lora_modules = config.num_hidden_layers * len(self.target.lora_target_modules)
        actual_lora = sum(1 for n, _ in model.named_modules() if "lora_A" in n and "default" in n)
        if actual_lora != expected_lora_modules:
            raise ValueError(
                f"LoRA injection mismatch: expected {expected_lora_modules} "
                f"lora_A modules, got {actual_lora}. Check target_modules "
                f"against the model architecture ({arch})."
            )
        model = finetune(model, tokenizer, self.target, dataset_name=dataset_name)

        print("\n[5/5] Exporting compressed model...")
        # BUG 2 fix: merge LoRA into base weights before saving
        merge_and_save_lora(model, tokenizer, output_dir)

        # Verify the output contains real model weights, not just adapters
        saved_files = os.listdir(output_dir)
        has_model = any(f.startswith("model") and f.endswith(".safetensors") for f in saved_files)
        has_only_adapter = all(f.startswith("adapter") for f in saved_files if f.endswith(".safetensors"))
        if not has_model or has_only_adapter:
            raise RuntimeError(
                f"Export failed: output directory {output_dir} does not "
                f"contain merged model weights. Files: {saved_files}"
            )

        # Optional GGUF / ONNX export
        if export_format == "gguf":
            gguf_path = os.path.join(output_dir, "model.gguf")
            export_gguf(output_dir, gguf_path)
        elif export_format == "onnx":
            onnx_dir = os.path.join(output_dir, "onnx")
            export_onnx(output_dir, onnx_dir)

        print(f"      Saved to: {output_dir}")
        print("\nDone.")
        return output_dir

    def dry_run(self, model_path: str) -> dict:
        """Print the compression plan without loading weights. Returns in <1s."""
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = getattr(config, "architectures", [None])[0]
        hidden = config.hidden_size
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        intermediate = config.intermediate_size
        layers = config.num_hidden_layers
        vocab = config.vocab_size
        head_dim = hidden // heads
        tie = getattr(config, "tie_word_embeddings", True)

        profile = {
            "total_params": 0,  # placeholder, not needed for planning
            "hidden_size": hidden,
            "num_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "architecture": arch,
            "tie_word_embeddings": tie,
        }

        # Estimate original params
        orig_estimated = total_params_at(profile, hidden, intermediate)
        profile["total_params"] = orig_estimated

        target_params = self._resolve_target_params(profile)
        new_h, new_i = self._plan_sizes(profile, target_params)
        estimated = total_params_at(profile, new_h, new_i)
        new_heads = new_h // head_dim
        compression_ratio = orig_estimated / max(estimated, 1)
        bytes_per_param = 2 if self.target.torch_dtype in ("float16", "bfloat16") else 4
        estimated_size_gb = estimated * bytes_per_param / 1e9

        result = {
            "architecture": arch,
            "original_params": orig_estimated,
            "target_params": target_params,
            "new_hidden_size": new_h,
            "new_intermediate_size": new_i,
            "new_num_heads": new_heads,
            "estimated_final_params": estimated,
            "estimated_size_gb": estimated_size_gb,
            "compression_ratio": compression_ratio,
            "head_dim": head_dim,
        }

        print("=" * 60)
        print("Qunart — Dry Run (Plan Only)")
        print("=" * 60)
        print(f"  Architecture:        {arch}")
        print(f"  Original params:     {orig_estimated:,}")
        print(f"  Target params:       {target_params:,}")
        print(f"  New hidden_size:     {new_h}")
        print(f"  New intermediate:    {new_i}")
        print(f"  New num_heads:       {new_heads}")
        print(f"  head_dim (constant): {head_dim}")
        print(f"  Estimated final:     {estimated:,}")
        print(f"  Estimated size:      {estimated_size_gb:.3f} GB")
        print(f"  Compression ratio:   {compression_ratio:.2f}x")
        return result

    def _resolve_target_params(self, profile: dict) -> int:
        if self.target.target_params is not None:
            return self.target.target_params
        if self.target.target_size_gb is not None:
            bytes_per_param = 2 if self.target.torch_dtype in ("float16", "bfloat16") else 4
            return int(self.target.target_size_gb * 1e9 / bytes_per_param)
        # default: 50% reduction
        return int(profile["total_params"] * 0.5)

    def _plan_sizes(self, profile: dict, target_params: int) -> Tuple[int, int]:
        """
        Binary search a width scaling factor s so that the pruned model
        lands close to target_params. We keep head_dim constant.
        """
        old_h = profile["hidden_size"]
        old_i = profile["intermediate_size"]
        head_dim = profile["head_dim"]
        if head_dim == 0:
            raise ValueError("Cannot determine head_dim; architecture unsupported.")

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
        return new_h, new_i



