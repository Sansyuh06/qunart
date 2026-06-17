===== ./README.md =====
# Qunart v2 — Universal LLM Compression Framework

## Vision

A single framework where a user uploads any Hugging Face model (or a local `safetensors`/`pytorch_model.bin` checkpoint) and Qunart automatically compresses it to a target parameter budget with minimal quality loss.

> Example: upload a 7B-parameter Llama/Qwen model and get a ~3.4B-parameter model that still behaves like the original 7B model, verified by perplexity and downstream task benchmarks.

## Core pipeline

```
Upload → Profile → Plan → Prune → Recover → Export
```

1. **Upload / Load** — `loader.py` loads any HF model + tokenizer.
2. **Profile** — `profiler.py` counts parameters and infers the model family.
3. **Plan** — `pipeline.py` chooses a target size and computes the required hidden/intermediate sizes.
4. **Prune** — `pruner.py` applies architecture-specific **structured width pruning**.
5. **Recover** — `recover.py` attaches LoRA/QLoRA and fine-tunes on a small corpus to restore quality.
6. **Export** — `exporter.py` saves the compressed model to `safetensors`, ONNX, or GGUF.

## Supported families (initially)

* Llama / Llama-2 / Llama-3 (`LlamaForCausalLM`)
* Qwen2 (`Qwen2ForCausalLM`)
* Mistral (`MistralForCausalLM`)
* Microsoft Phi-3 (`Phi3ForCausalLM`)

Other families (Gemma, Phi-3, GPT-2) can be added by writing a new `Pruner` class and registering it in `pipeline.py`.

## Quick start

```bash
pip install -r requirements.txt

# CLI example: compress Llama-2-7B to ~3.4B parameters
python cli.py \
  --model meta-llama/Llama-2-7b-hf \
  --output ./llama-3.4b \
  --target-params 3400000000 \
  --finetune-steps 500 \
  --lora-r 16 \
  --lora-alpha 32
```

```python
# or in Python
from qunart import CompressionTarget, CompressionPipeline

target = CompressionTarget(
    target_params=3_400_000_000,
    finetune_steps=500,
    lora_r=16,
    lora_alpha=32,
)

pipeline = CompressionPipeline(target)
pipeline.run('meta-llama/Llama-2-7b-hf', './llama-3.4b')
```

## How the compression works

For Llama-style models, the most reliable way to cut parameters in half is **structured width pruning** while keeping the **head dimension constant**:

* `hidden_size` is reduced from `H` to `H'`.
* `intermediate_size` is reduced from `I` to `I'`.
* `num_attention_heads` and `num_key_value_heads` are reduced proportionally so `head_dim = H / num_heads` stays the same.
* Because `head_dim` does not change, the rotary embeddings can be reused as-is.
* All projection weights are sliced in place, the config is updated, and the model is saved as a smaller checkpoint.

This is more aggressive than simple head pruning but safer than depth pruning because every layer keeps its full residual path.

### QUBO component selection

The framework also uses a QUBO/greedy solver (`qubo.py`) to choose **which MLP neurons to keep** per layer based on weight importance. The cardinality constraint ("keep exactly K neurons") is encoded as a QUBO penalty term and solved with `dwave-neal` when available, falling back to a greedy solver.

## Quality recovery

After pruning, the model is wrapped with LoRA adapters (PEFT) and fine-tuned on a small instruction corpus (default: `yahma/alpaca-cleaned`). This is the standard recovery recipe for compressed LLMs and is the main reason a 3.4B model can stay close to the original 7B model.

## Architecture of the code

```
qunart_v2_framework/
├── qunart/
│   ├── config.py      # CompressionTarget dataclass
│   ├── loader.py      # AutoModel loading
│   ├── profiler.py    # Parameter accounting
│   ├── importance.py  # Per-weight / per-head importance
│   ├── qubo.py        # QUBO solver (greedy + dwave-neal fallback)
│   ├── pruner.py      # Architecture-specific pruners
│   ├── recover.py     # LoRA + fine-tune
│   ├── exporter.py    # Save compressed model
│   └── pipeline.py    # End-to-end orchestration
├── cli.py             # Command-line entry point
├── example.py         # Minimal Python example
└── requirements.txt
```

## Roadmap

- [x] Generic HF loader and profiler
- [x] Llama/Qwen/Mistral width pruner with constant head_dim
- [x] QUBO-based MLP neuron selection
- [x] LoRA recovery pipeline
- [x] Phi-3 pruner (`Phi3ForCausalLM`)
- [ ] Add pruners for Gemma, Phi-3.5/Phi-4, GPT-2
- [ ] Attention-head QUBO selection (non-contiguous keep set)
- [ ] Quantization-aware pruning (INT8 / INT4)
- [ ] Export to ONNX and GGUF
- [ ] Add `lm-evaluation-harness` evaluation stage
- [ ] Web UI / API for upload → compress → download

## Important caveats

* This is a framework **scaffold**. The width-pruning recipe is sound but the exact quality/size trade-off depends on the model and the recovery data. Always evaluate before shipping.
* The `dwave-neal` QUBO solver is optional. Without it, the framework falls back to greedy selection.
* Fine-tuning a 7B model still needs a GPU with at least ~16 GB VRAM (or use QLoRA / `bitsandbytes` for 8-bit optimizers).
* The CLI currently does not automatically export to ONNX/GGUF; use `optimum-cli` or `llama.cpp` converters on the saved checkpoint.








