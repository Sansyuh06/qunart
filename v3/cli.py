import argparse

from qunart import CompressionPipeline, CompressionTarget


def main():
    parser = argparse.ArgumentParser(
        description="Qunart v2: compress any Hugging Face LLM to a target size."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--output", default=None, help="Directory to save compressed model"
    )
    parser.add_argument(
        "--target-params", type=int, default=None, help="Target parameter count"
    )
    parser.add_argument(
        "--target-size-gb", type=float, default=None, help="Target size in GB (fp16/bf16)"
    )
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print compression plan and exit"
    )
    parser.add_argument(
        "--export", choices=["hf", "gguf", "onnx"], default="hf", help="Export format"
    )
    parser.add_argument(
        "--selection-method",
        choices=["greedy", "qubo"],
        default="greedy",
        help="Neuron selection method",
    )
    parser.add_argument(
        "--lambda-redundancy",
        type=float,
        default=0.1,
        help="Redundancy penalty weight for QUBO",
    )

    args = parser.parse_args()

    if args.target_params is None and args.target_size_gb is None:
        raise ValueError("Specify one of --target-params or --target-size-gb")

    target = CompressionTarget(
        target_params=args.target_params,
        target_size_gb=args.target_size_gb,
        finetune_steps=args.finetune_steps,
        finetune_lr=args.finetune_lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        torch_dtype=args.torch_dtype,
        device=args.device,
        recovery_dataset=args.dataset,
        selection_method=args.selection_method,
        lambda_redundancy=args.lambda_redundancy,
    )

    pipeline = CompressionPipeline(target)

    if args.dry_run:
        pipeline.dry_run(args.model)
        return

    if not args.output:
        raise ValueError("Specify --output directory")

    pipeline.run(
        args.model,
        args.output,
        dataset_name=args.dataset,
        export_format=args.export,
    )


if __name__ == "__main__":
    main()
