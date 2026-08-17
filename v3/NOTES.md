# qunart Technical Notes & Architecture Log

## 1. Hard Gate Status & Test Summary
- **Test Suite**: 24 tests passed across 4 test modules:
  - `tests/test_shapes.py`: 12 passed (Llama & Phi-3 shape slicing, forward pass, autoregressive token generation, constant `head_dim`, GQA grouping ratio preservation, and embedding bias guard).
  - `tests/test_export.py`: 2 passed (`merge_and_save_lora` saves actual `model.safetensors` base weights + `config.json`, successfully reloaded via `AutoModelForCausalLM.from_pretrained`).
  - `tests/test_lora_targets.py`: 3 passed (exact LoRA adapter injection counts for standard Llama projections and Phi-3 fused `qkv_proj`/`gate_up_proj` projections).
  - `tests/test_selection.py`: 7 passed (greedy top-K correctness, QUBO simulated annealing K-subset validity, similarity matrix computation, large-$N$ candidate prefiltering).

## 2. Bug Resolutions
- **BUG 1 (Fatal AttributeError)**: `nn.Embedding` has no `.bias` attribute. Guarded all `.bias` accesses across `LlamaWidthPruner` and `Phi3WidthPruner` using `getattr(module, "bias", None) is not None`.
- **BUG 2 (Fatal Adapter-only Export)**: Replaced `save_model` with `merge_and_save_lora` in `pipeline.py` to ensure `model.merge_and_unload()` bakes LoRA adapters into base weights before serializing. Added post-save assertion to verify `model*.safetensors` exists on disk.
- **BUG 3 (Fatal Dimension Slicing Mismatch)**: Sliced both intermediate dimension (`keep`) and hidden dimension (`: self.new_h`) in `_prune_mlp` and `_prune_phi3_mlp`. Added slicing for final RMSNorm/LayerNorm `model.model.norm`.
- **BUG 4 (Phi-3 LoRA Target Mismatch)**: Dynamically selected target module list based on detected architecture:
  - Llama/Qwen2/Mistral: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - Phi-3: `["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`
- **BUG 5 (Decorative QUBO)**: Upgraded `QUBOSolver` with genuine pairwise redundancy penalty ($S_{ij} = \text{cosine\_similarity}([\text{gate}_i ; \text{up}_i], [\text{gate}_j ; \text{up}_j])$), preventing correlated redundant neurons from being simultaneously selected.
- **BUG 6 (Estimator Calibration)**: Added post-prune parameter count verification asserting actual parameter count matches `total_params_at()` within 2%.
- **BUG 7 (Broken CI)**: Created `.github/workflows/test.yml` running pytest in clean matrix environments.

## 3. Hardware & Execution Constraints
- Host GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.6 GB VRAM).
- TinyLlama-1.1B-Chat fits comfortably within this VRAM budget for full forward/backward LoRA recovery.
- Phi-3-mini-4k-instruct requires $\ge 16$ GB VRAM for unquantized recovery training; its width pruner and shape slicing are verified offline via unit tests.
- On-device edge runtime (tok/s, Peak RAM) marked as TBD pending deployment on target hardware as specified in `docs/DEPLOY_ANDROID.md`.

## 4. GGUF Export & Quantization Constraints
- llama.cpp requires matrix dimensions (`hidden_size`, `intermediate_size`) to align with tensor block boundaries (multiples of 64).
- Attention heads must divide evenly into hidden size to keep `head_dim` constant.
- For non-standard width factors, the planning stage aligns dimensions to multiples of 64.

## 5. SliceGPT-Style Orthogonal Rotation Design (Optional Stretch)
### Concept
Channel $k$ in transformer residual streams is a coordinate in an arbitrary basis. Simple coordinate slicing drops arbitrary basis directions.
By inserting orthogonal matrices $Q \in O(d)$ between blocks:
1. Compute activation covariance $C = \frac{1}{N}\sum x x^T$ on a calibration set (e.g. 128 Alpaca sequences).
2. Perform eigenvalue decomposition $C = Q \Lambda Q^T$, ordering eigenvectors by decreasing variance (PCA basis).
3. Transform layer weights: $W_{\text{in}} \leftarrow Q^T W_{\text{in}}$ and $W_{\text{out}} \leftarrow W_{\text{out}} Q$.
4. Slicing the tail of the rotated hidden state preserves the highest-energy principal components, yielding lower initial perplexity before LoRA recovery.
