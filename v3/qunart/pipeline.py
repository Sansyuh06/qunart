from typing import Optional, Tuple

from .config import CompressionTarget
from .loader import ModelLoader
from .profiler import profile_model, total_params_at
from .pruner import LlamaWidthPruner, Phi3WidthPruner
from .recover import attach_lora, finetune
from .exporter import save_model


SUPPORTED = (
    LlamaWidthPruner.SUPPORTED_ARCHS
    + Phi3WidthPruner.SUPPORTED_ARCHS
)


class CompressionPipeline:
    """End-to-end: load → profile → plan → prune → recover → export."""

    def __init__(self, target: CompressionTarget):
        self.target = target
        self.loader = ModelLoader(
            device=target.device, torch_dtype=target.torch_dtype
        )

    def run(self, model_path: str, output_dir: str, dataset_name: Optional[str] = None):
        print("=" * 60)
        print("Qunart v2 — Universal LLM Compression")
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
        if arch in LlamaWidthPruner.SUPPORTED_ARCHS:
            pruner = LlamaWidthPruner(new_h, new_i)
        elif arch in Phi3WidthPruner.SUPPORTED_ARCHS:
            pruner = Phi3WidthPruner(new_h, new_i)
        else:
            raise ValueError(
                f"Architecture '{arch}' is not yet supported. "
                f"Supported: {SUPPORTED}"
            )
        model, config = pruner.prune(model, config)

        print("\n[4/5] Recovery fine-tuning (LoRA)...")
        model = attach_lora(model, self.target)
        model = finetune(model, tokenizer, self.target, dataset_name=dataset_name)

        print("\n[5/5] Exporting compressed model...")
        save_model(model, tokenizer, output_dir)
        print(f"      Saved to: {output_dir}")

        print("\nDone.")
        return output_dir

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


