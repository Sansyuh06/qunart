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
      1. Run llama.cpp's convert_hf_to_gguf.py to get an f16 GGUF.
      2. Run llama-quantize to apply the requested quantisation.

    If llama_cpp_dir is None, we look for llama.cpp on PATH and in
    common locations.
    """
    # Find llama.cpp tools
    converter = _find_converter(llama_cpp_dir)
    quantizer = _find_quantizer(llama_cpp_dir)

    f16_path = out_path.replace(".gguf", ".f16.gguf")

    # Step 1: convert HF → f16 GGUF
    print(f"  Converting HF model to GGUF (f16)...")
    cmd_convert = ["python", converter, model_dir, "--outfile", f16_path, "--outtype", "f16"]
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed:\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    print(f"  f16 GGUF written to: {f16_path}")

    # Step 2: quantise
    if quant.upper() != "F16":
        print(f"  Quantising to {quant}...")
        cmd_quant = [quantizer, f16_path, out_path, quant]
        result = subprocess.run(cmd_quant, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"GGUF quantisation failed:\nstdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )
        # Clean up the intermediate f16 file
        if os.path.exists(f16_path) and os.path.exists(out_path):
            os.remove(f16_path)
        print(f"  Quantised GGUF written to: {out_path}")
    else:
        if f16_path != out_path:
            shutil.move(f16_path, out_path)
        print(f"  GGUF written to: {out_path}")

    return out_path


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
    # Check common locations
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
