import os
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    safe_serialization: bool = True,
):
    """Save the pruned/recovered model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def merge_and_save_lora(model, tokenizer, output_dir: str):
    """Merge LoRA adapters into base weights and save."""
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    return save_model(model, tokenizer, output_dir)
