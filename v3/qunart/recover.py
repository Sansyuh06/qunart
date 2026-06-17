from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import CompressionTarget


def attach_lora(model, target: CompressionTarget):
    """Wrap the pruned model with LoRA adapters for recovery fine-tuning."""
    peft_config = LoraConfig(
        r=target.lora_r,
        lora_alpha=target.lora_alpha,
        target_modules=target.lora_target_modules,
        lora_dropout=target.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


def _format_alpaca(example: dict) -> dict:
    if example.get("input"):
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


def finetune(
    model,
    tokenizer,
    target: CompressionTarget,
    dataset_name: Optional[str] = None,
):
    """Lightweight recovery fine-tune on an instruction-style corpus."""
    dataset_name = dataset_name or target.recovery_dataset

    print(f"Loading recovery dataset: {dataset_name} ...")
    ds = load_dataset(dataset_name, split=target.recovery_dataset_split)
    n = min(target.recovery_samples, len(ds))
    ds = ds.select(range(n))
    ds = ds.map(_format_alpaca, remove_columns=ds.column_names)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=target.max_seq_length,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    fp16 = torch.cuda.is_available() and target.torch_dtype in ("float16", "bfloat16")
    bf16 = torch.cuda.is_available() and target.torch_dtype == "bfloat16"

    args = TrainingArguments(
        output_dir="./qunart_recovery",
        num_train_epochs=1,
        max_steps=target.finetune_steps,
        per_device_train_batch_size=target.batch_size,
        gradient_accumulation_steps=target.gradient_accumulation_steps,
        learning_rate=target.finetune_lr,
        logging_steps=max(10, target.finetune_steps // 20),
        save_strategy="no",
        fp16=fp16 and not bf16,
        bf16=bf16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    return model

