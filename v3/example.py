"""Minimal Python example: compress a 7B Llama-style model to ~3.4B params."""

from qunart import CompressionPipeline, CompressionTarget

if __name__ == "__main__":
    target = CompressionTarget(
        target_params=3_400_000_000,  # 3.4B
        finetune_steps=500,
        lora_r=16,
        lora_alpha=32,
        torch_dtype="float16",
        device="auto",
    )

    pipeline = CompressionPipeline(target)
    pipeline.run(
        model_path="meta-llama/Llama-2-7b-hf",
        output_dir="./llama-3.4b-compressed",
    )
