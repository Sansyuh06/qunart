from dataclasses import dataclass, field
from typing import Optional, Literal, List


@dataclass
class CompressionTarget:
    """User-defined compression goal and recovery hyperparameters."""

    # --- target ---
    target_params: Optional[int] = None
    target_size_gb: Optional[float] = None

    # --- quantization (post-recovery) ---
    quantization: Optional[Literal["int8", "int4", "none"]] = "none"

    # --- quality / recovery ---
    quality_retention: float = 0.95
    finetune_steps: int = 500
    finetune_lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # --- training ---
    max_seq_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    recovery_dataset: str = "yahma/alpaca-cleaned"
    recovery_dataset_split: str = "train"
    recovery_samples: int = 2000

    # --- infra ---
    device: str = "auto"
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    architecture: Optional[str] = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
