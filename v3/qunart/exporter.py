import os
import subprocess
import shutil
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
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    return output_dir


def merge_and_save_lora(model, tokenizer, output_dir: str):
    """Merge LoRA adapters into base weights and save."""
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    return save_model(model, tokenizer, output_dir)


def export_gguf(
    model_dir: str,
    out_path: str,
    quant: str = "Q4_K_M",
    llama_cpp_dir: Optional[str] = None,
):
    """
    Convert an HF model directory to a quantised GGUF file.

    Steps:
      1. Converts HF weights to GGUF using native gguf.GGUFWriter (or convert_hf_to_gguf.py).
      2. Quantises using llama-quantize if available, or exports native GGUF.
    """
    # Ensure directory exists for out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    f16_path = out_path if quant.upper() == "F16" else out_path.replace(".gguf", ".f16.gguf")

    # Step 1: Export to GGUF
    print(f"  Converting HF model to GGUF...")
    try:
        _export_gguf_native(model_dir, f16_path)
        print(f"  GGUF written to: {f16_path}")
    except Exception as exc:
        # Fallback to llama.cpp converter script if native export encounters an unsupported arch
        converter = _find_converter(llama_cpp_dir)
        cmd_convert = ["python", converter, model_dir, "--outfile", f16_path, "--outtype", "f16"]
        result = subprocess.run(cmd_convert, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"GGUF conversion failed:\nstdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        print(f"  GGUF written via converter to: {f16_path}")

    # Step 2: Quantise if requested and quantizer is available
    if quant.upper() != "F16":
        try:
            quantizer = _find_quantizer(llama_cpp_dir)
            print(f"  Quantising to {quant} via llama-quantize...")
            cmd_quant = [quantizer, f16_path, out_path, quant]
            result = subprocess.run(cmd_quant, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"GGUF quantisation failed:\nstdout: {result.stdout[-2000:]}\n"
                    f"stderr: {result.stderr[-2000:]}"
                )
            if os.path.exists(f16_path) and f16_path != out_path:
                os.remove(f16_path)
            print(f"  Quantised GGUF written to: {out_path}")
        except FileNotFoundError:
            # llama-quantize not built yet; retain the native GGUF file
            if f16_path != out_path:
                shutil.move(f16_path, out_path)
            print(f"  Note: llama-quantize not found; saved base GGUF to: {out_path}")
    else:
        if f16_path != out_path and os.path.exists(f16_path):
            shutil.move(f16_path, out_path)

    return out_path


def _export_gguf_native(model_dir: str, out_path: str):
    """Native Python GGUF export for Llama and Phi-3 models using the gguf package."""
    import json
    import torch
    from safetensors.torch import load_file
    import gguf

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    arch = config.get("architectures", ["LlamaForCausalLM"])[0]
    is_phi = "Phi3" in arch or "Phi" in arch

    writer = gguf.GGUFWriter(out_path, "phi3" if is_phi else "llama")

    # Add model metadata
    writer.add_name(config.get("_name_or_path", "qunart-compressed-model"))
    writer.add_uint32("llama.block_count" if not is_phi else "phi3.block_count", config["num_hidden_layers"])
    writer.add_uint32("llama.context_length" if not is_phi else "phi3.context_length", config.get("max_position_embeddings", 2048))
    writer.add_uint32("llama.embedding_length" if not is_phi else "phi3.embedding_length", config["hidden_size"])
    writer.add_uint32("llama.feed_forward_length" if not is_phi else "phi3.feed_forward_length", config["intermediate_size"])
    writer.add_uint32("llama.attention.head_count" if not is_phi else "phi3.attention.head_count", config["num_attention_heads"])
    writer.add_uint32("llama.attention.head_count_kv" if not is_phi else "phi3.attention.head_count_kv", config.get("num_key_value_heads", config["num_attention_heads"]))
    writer.add_float32("llama.attention.layer_norm_rms_epsilon" if not is_phi else "phi3.attention.layer_norm_rms_epsilon", config.get("rms_norm_eps", 1e-5))

    head_dim = config["hidden_size"] // config["num_attention_heads"]
    writer.add_uint32("llama.rope.dimension_count" if not is_phi else "phi3.rope.dimension_count", head_dim)
    writer.add_float32("llama.rope.freq_base" if not is_phi else "phi3.rope.freq_base", config.get("rope_theta", 10000.0))

    # Add tokenizer if available
    tok_json = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        try:
            with open(tok_json, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
                vocab = tok_data.get("model", {}).get("vocab", {})
                if vocab:
                    tokens = [b""] * len(vocab)
                    scores = [0.0] * len(vocab)
                    tok_types = [1] * len(vocab)
                    for token_str, idx in vocab.items():
                        if idx < len(tokens):
                            tokens[idx] = token_str.encode("utf-8")
                    writer.add_tokenizer_model("llama")
                    writer.add_token_list(tokens)
                    writer.add_token_scores(scores)
                    writer.add_token_types(tok_types)
        except Exception:
            pass

    # Load model state dict
    state_dict = {}
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(model_dir, fname)))
        elif fname.endswith(".bin") and "pytorch_model" in fname:
            state_dict.update(torch.load(os.path.join(model_dir, fname), map_location="cpu"))

    # Map tensor names
    tensor_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for i in range(config["num_hidden_layers"]):
        tensor_map[f"model.layers.{i}.input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
        tensor_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
        tensor_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
        tensor_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
        tensor_map[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"
        tensor_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"
        tensor_map[f"model.layers.{i}.mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
        tensor_map[f"model.layers.{i}.mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
        tensor_map[f"model.layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

    for hf_name, tensor in state_dict.items():
        gguf_name = tensor_map.get(hf_name, hf_name)
        arr = tensor.detach().cpu().to(torch.float16).numpy()
        writer.add_tensor(gguf_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def export_onnx(model_dir: str, out_dir: str):
    """Export an HF model to ONNX via optimum-cli."""
    print(f"  Exporting to ONNX...")
    cmd = [
        "python", "-m", "optimum.exporters.onnx",
        "--model", model_dir,
        "--task", "causal-lm",
        out_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ONNX export failed:\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    print(f"  ONNX model written to: {out_dir}")
    return out_dir


def _find_converter(llama_cpp_dir: Optional[str] = None) -> str:
    """Locate convert_hf_to_gguf.py."""
    candidates = []
    if llama_cpp_dir:
        candidates.append(os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"))
    # Check bundled scripts directory and common locations
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "convert_hf_to_gguf.py"))
    candidates.append(os.path.join("scripts", "convert_hf_to_gguf.py"))
    for d in ["llama.cpp", "../llama.cpp", os.path.expanduser("~/llama.cpp")]:
        candidates.append(os.path.join(d, "convert_hf_to_gguf.py"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Cannot find convert_hf_to_gguf.py. Either:\n"
        "  1. Clone llama.cpp next to this repo, or\n"
        "  2. Pass llama_cpp_dir='/path/to/llama.cpp' explicitly.\n"
        "  See docs/DEPLOY_ANDROID.md for setup instructions."
    )


def _find_quantizer(llama_cpp_dir: Optional[str] = None) -> str:
    """Locate the llama-quantize binary."""
    # Check PATH first
    q = shutil.which("llama-quantize")
    if q:
        return q
    candidates = []
    if llama_cpp_dir:
        candidates.append(os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize"))
        candidates.append(os.path.join(llama_cpp_dir, "llama-quantize"))
    for d in ["llama.cpp", "../llama.cpp", os.path.expanduser("~/llama.cpp")]:
        candidates.append(os.path.join(d, "build", "bin", "llama-quantize"))
        candidates.append(os.path.join(d, "llama-quantize"))
    for c in candidates:
        if os.path.isfile(c):
            return c
        # Windows: check .exe
        if os.path.isfile(c + ".exe"):
            return c + ".exe"
    raise FileNotFoundError(
        "Cannot find llama-quantize binary. Build llama.cpp first:\n"
        "  cd llama.cpp && cmake -B build && cmake --build build\n"
        "  See docs/DEPLOY_ANDROID.md for setup instructions."
    )
