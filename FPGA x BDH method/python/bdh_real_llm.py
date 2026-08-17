import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdh_bpe_tokenizer import BDHBPETokenizer, get_tinystories_corpus


CHECKPOINT_PATH = Path(__file__).parent / "bdh_llm_checkpoint.pt"

BDHTokenizer = BDHBPETokenizer


class DotProductEngine(Protocol):
    zero_skip: bool

    def compute_dot_product(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        sparse_mode: bool = True,
    ) -> int:
        ...


@dataclass(frozen=True)
class TinyTokenFrame:
    token: str
    execution_mode: str
    zero_sparsity_pct: float
    active_pct: float
    fpga_speedup: float
    dsp_ops: int
    accumulator: int
    uart_bytes_sent: int
    graph_nodes: List[dict[str, object]]
    graph_edges: List[dict[str, object]]
    synaptic_matrix_16x16: List[float]


class PyTorchBDHModel(nn.Module):
    """Real Baby Dragon Hatchling (BDH) PyTorch Language Model."""

    def __init__(
        self, vocab_size: int = 384, embed_dim: int = 128, ff_dim: int = 256
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)

        # BDH Query, Key, Value Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Feedforward Block
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _quantize(values: np.ndarray) -> Tuple[np.ndarray, float]:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        if max_abs == 0.0:
            return np.zeros_like(values, dtype=np.int8), 1.0
        scale = 127.0 / max_abs
        quantized = np.clip(np.rint(values * scale), -128, 127)
        return quantized.astype(np.int8), scale

    def _dispatch_linear(
        self,
        module: nn.Linear,
        x: torch.Tensor,
        engine: Optional[DotProductEngine],
        pre_activations: List[np.ndarray],
        accumulators: List[int],
        dsp_ops_list: List[int],
        execution_modes: List[str],
        is_head: bool = False,
        offload_mode: str = "full",
        hardware_required: bool = False,
    ) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        x_i8, _ = self._quantize(x_np)
        pre_activations.append(x_i8)

        if engine is None:
            if hardware_required:
                raise RuntimeError(
                    "FPGA hardware execution required, but Tang Nano 9K is "
                    "disconnected or unvalidated!"
                )
            execution_modes.append("CPU (HOST)")
            return module(x)

        if offload_mode == "head" and not is_head:
            execution_modes.append("CPU (HOST)")
            return module(x)

        weights_np = module.weight.detach().cpu().numpy()
        bias_np = (
            module.bias.detach().cpu().numpy()
            if module.bias is not None
            else None
        )
        weights_i8, w_scale = self._quantize(weights_np)
        sparse_mode = getattr(engine, "zero_skip", True)

        batch_size, seq_len, in_dim = x.shape
        out_dim = module.out_features

        if hasattr(engine, "compute_layer_projection"):
            try:
                # In autoregressive inference, only the active newest timestep row (x[:, -1:, :])
                # needs to be projected over UART to keep per-token latency at ~0.15s
                if seq_len > 1:
                    x_active_np = x_np[:, -1:, :].reshape(-1, in_dim)
                    x_active_i8, x_scale = self._quantize(x_active_np)
                    dequant_scale = (w_scale * x_scale) or 1.0

                    acc_matrix_active = engine.compute_layer_projection(
                        weights_i8, x_active_i8, sparse_mode=sparse_mode
                    )
                    accumulators.append(int(np.sum(acc_matrix_active)))
                    dsp_ops_list.append(batch_size * 1 * in_dim * out_dim)

                    out_active_fp32 = acc_matrix_active.astype(np.float32) / dequant_scale
                    if bias_np is not None:
                        out_active_fp32 += bias_np

                    y_prefix = module(x[:, :-1, :])
                    y_active = torch.from_numpy(out_active_fp32).reshape(
                        batch_size, 1, out_dim
                    ).to(device=x.device, dtype=torch.float32)
                    y = torch.cat([y_prefix, y_active], dim=1)
                    execution_modes.append("FPGA")
                    return y
                else:
                    x_flat_np = x_np.reshape(-1, in_dim)
                    x_flat_i8, x_scale = self._quantize(x_flat_np)
                    dequant_scale = (w_scale * x_scale) or 1.0

                    acc_matrix = engine.compute_layer_projection(
                        weights_i8, x_flat_i8, sparse_mode=sparse_mode
                    )
                    accumulators.append(int(np.sum(acc_matrix)))
                    dsp_ops_list.append(batch_size * seq_len * in_dim * out_dim)

                    out_fp32 = acc_matrix.astype(np.float32) / dequant_scale
                    if bias_np is not None:
                        out_fp32 += bias_np
                    y = torch.from_numpy(out_fp32).reshape(
                        batch_size, seq_len, out_dim
                    )
                    execution_modes.append("FPGA")
                    return y.to(device=x.device, dtype=torch.float32)
            except Exception as exc:
                if hardware_required:
                    raise RuntimeError(
                        f"FPGA hardware execution failed on {module}: {exc}"
                    ) from exc
                execution_modes.append("CPU FALLBACK")
                return module(x)

        if hasattr(engine, "compute_dot_product"):
            try:
                batch_outputs = []
                for b in range(batch_size):
                    seq_outputs = []
                    for t in range(seq_len):
                        x_t_np = x_np[b, t]
                        x_t_i8, x_scale = self._quantize(x_t_np)
                        dequant_scale = w_scale * x_scale
                        if dequant_scale == 0.0:
                            dequant_scale = 1.0

                        row_outputs = []
                        for i in range(out_dim):
                            w_row_i8 = weights_i8[i]
                            acc = engine.compute_dot_product(
                                w_row_i8, x_t_i8, sparse_mode=sparse_mode
                            )
                            accumulators.append(acc)
                            val = float(acc) / dequant_scale
                            if bias_np is not None:
                                val += float(bias_np[i])
                            row_outputs.append(val)
                        dsp_ops_list.append(in_dim * out_dim)
                        seq_outputs.append(row_outputs)
                    batch_outputs.append(seq_outputs)

                execution_modes.append("FPGA")
                return torch.tensor(
                    batch_outputs, dtype=torch.float32, device=x.device
                )
            except Exception as exc:
                if hardware_required:
                    raise RuntimeError(
                        f"FPGA hardware dot-product failed: {exc}"
                    ) from exc
                execution_modes.append("CPU FALLBACK")
                return module(x)

        execution_modes.append("CPU (HOST)")
        return module(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        engine: Optional[DotProductEngine] = None,
        offload_mode: str = "full",
        hardware_required: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray], int, int, str]:
        batch_size, seq_len = input_ids.shape
        tok_e = self.tok_embed(input_ids)
        pos_e = self.pos_embed[:, :seq_len, :]
        x = tok_e + pos_e

        pre_activations: List[np.ndarray] = []
        accumulators: List[int] = []
        dsp_ops_list: List[int] = []
        execution_modes: List[str] = []

        # Softmax-free Causal Hebbian Fast-Weight Attention
        Q = F.relu(
            self._dispatch_linear(
                self.q_proj,
                x,
                engine,
                pre_activations,
                accumulators,
                dsp_ops_list,
                execution_modes,
                is_head=False,
                offload_mode=offload_mode,
                hardware_required=hardware_required,
            )
        )
        K = F.relu(
            self._dispatch_linear(
                self.k_proj,
                x,
                engine,
                pre_activations,
                accumulators,
                dsp_ops_list,
                execution_modes,
                is_head=False,
                offload_mode=offload_mode,
                hardware_required=hardware_required,
            )
        )
        V = self._dispatch_linear(
            self.v_proj,
            x,
            engine,
            pre_activations,
            accumulators,
            dsp_ops_list,
            execution_modes,
            is_head=False,
            offload_mode=offload_mode,
            hardware_required=hardware_required,
        )

        # Causal fast weight accumulation rho_t = rho_{t-1} + K_t^T V_t
        attn_out_list = []
        running_rho = torch.zeros(
            batch_size,
            self.embed_dim,
            self.embed_dim,
            device=input_ids.device,
        )
        scale = 1.0 / math.sqrt(self.embed_dim)
        for t in range(seq_len):
            kt = K[:, t:t + 1, :].transpose(1, 2)
            vt = V[:, t:t + 1, :]
            running_rho = running_rho + torch.bmm(kt, vt) * scale
            qt = Q[:, t:t + 1, :]
            attn_out_list.append(torch.bmm(qt, running_rho))

        attn_out = torch.cat(attn_out_list, dim=1)
        rho = running_rho

        out_proj_val = F.relu(
            self._dispatch_linear(
                self.out_proj,
                attn_out,
                engine,
                pre_activations,
                accumulators,
                dsp_ops_list,
                execution_modes,
                is_head=False,
                offload_mode=offload_mode,
                hardware_required=hardware_required,
            )
        )
        x = x + out_proj_val

        # FF block
        hidden = F.relu(
            self._dispatch_linear(
                self.fc1,
                x,
                engine,
                pre_activations,
                accumulators,
                dsp_ops_list,
                execution_modes,
                is_head=False,
                offload_mode=offload_mode,
                hardware_required=hardware_required,
            )
        )
        fc2_val = F.relu(
            self._dispatch_linear(
                self.fc2,
                hidden,
                engine,
                pre_activations,
                accumulators,
                dsp_ops_list,
                execution_modes,
                is_head=False,
                offload_mode=offload_mode,
                hardware_required=hardware_required,
            )
        )
        x = x + fc2_val

        logits = self._dispatch_linear(
            self.head,
            x,
            engine,
            pre_activations,
            accumulators,
            dsp_ops_list,
            execution_modes,
            is_head=True,
            offload_mode=offload_mode,
            hardware_required=hardware_required,
        )
        total_acc = sum(accumulators) if accumulators else 0
        total_dsp = (
            sum(dsp_ops_list)
            if dsp_ops_list
            else (
                seq_len
                * (
                    4 * (self.embed_dim * self.embed_dim)
                    + self.embed_dim * self.ff_dim
                    + self.ff_dim * self.embed_dim
                    + self.embed_dim * self.vocab_size
                )
            )
        )

        if "CPU FALLBACK" in execution_modes:
            mode_str = "CPU FALLBACK"
        elif "FPGA" in execution_modes:
            if offload_mode == "head":
                mode_str = "FPGA (HEAD-ONLY)"
            else:
                mode_str = "FPGA (FULL)"
        else:
            mode_str = "CPU (HOST)"

        return logits, rho, pre_activations, total_acc, total_dsp, mode_str


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RealBDHLanguageModel:
    """Real BDH LM with hardware acceleration & zero-skipping telemetry."""

    def __init__(self, embed_dim: int = 128, ff_dim: int = 256) -> None:
        self.tokenizer = BDHBPETokenizer()
        self.device = get_device()
        self.model = PyTorchBDHModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
        ).to(self.device)
        self.model.eval()

        if CHECKPOINT_PATH.exists():
            try:
                state_dict = torch.load(
                    CHECKPOINT_PATH,
                    map_location=self.device,
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict)
            except Exception as exc:
                print(
                    f"Checkpoint load failed ({exc}), retraining model...",
                    flush=True,
                )
                train_and_save_checkpoint(self.model, self.tokenizer)
        else:
            train_and_save_checkpoint(self.model, self.tokenizer)

    def generate(
        self,
        prompt: str,
        engine: Optional[DotProductEngine] = None,
        max_tokens: int = 60,
        offload_mode: str = "full",
        hardware_required: bool = False,
    ) -> List[TinyTokenFrame]:
        return list(
            self.iter_generate(
                prompt,
                engine=engine,
                max_tokens=max_tokens,
                offload_mode=offload_mode,
                hardware_required=hardware_required,
            )
        )

    def iter_generate(
        self,
        prompt: str,
        engine: Optional[DotProductEngine] = None,
        max_tokens: int = 60,
        temperature: float = 0.7,
        top_k: int = 3,
        top_p: float = 0.9,
        offload_mode: str = "full",
        hardware_required: bool = False,
    ) -> Iterator[TinyTokenFrame]:
        prompt_clean = prompt.strip().lower()
        if not prompt_clean.endswith(("?", ".", "!", ":")):
            prompt_clean = f"{prompt_clean}?"

        context_ids = self.tokenizer.encode(prompt_clean, add_bos=True)
        device = self.device
        synaptic_rho: List[float] = []

        generated_count = 0

        for _ in range(max_tokens):
            inp = torch.tensor([context_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                (
                    logits_seq,
                    rho_tensor,
                    pre_acts,
                    acc_sum,
                    dsp_ops,
                    exec_mode,
                ) = self.model(
                    inp,
                    engine=engine,
                    offload_mode=offload_mode,
                    hardware_required=hardware_required,
                )

            next_logits = logits_seq[0, -1, :].cpu()
            rho_matrix = rho_tensor[0].cpu().numpy()

            synaptic_rho = self._format_rho_16x16(rho_matrix)

            if pre_acts:
                all_acts = np.concatenate([a.ravel() for a in pre_acts])
                zero_sparsity = float(np.mean(all_acts == 0) * 100.0)
            else:
                zero_sparsity = 0.0
            active_pct = max(100.0 - zero_sparsity, 0.1)
            fpga_speedup = round(100.0 / active_pct, 2)

            # Apply repetition penalty to prevent repeated words/loops
            for prev_id in set(context_ids[-24:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= 1.25
                else:
                    next_logits[prev_id] *= 1.25

            if top_k == 1 or temperature <= 0.0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                scaled_logits = next_logits / max(temperature, 1e-5)
                if top_k > 0 and top_k < scaled_logits.size(-1):
                    top_k_val = min(top_k, scaled_logits.size(-1))
                    min_topk = torch.topk(scaled_logits, top_k_val).values[-1]
                    scaled_logits[scaled_logits < min_topk] = -float("inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        scaled_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[..., :-1].clone()
                    )
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[
                        sorted_indices_to_remove
                    ]
                    scaled_logits[indices_to_remove] = -float("inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            context_ids.append(next_id)
            token_char = self.tokenizer.decode([next_id])
            if generated_count == 0 and token_char in ("?", ":", " "):
                token_char = ""
            generated_count += 1

            if pre_acts and engine is not None and "FPGA" in exec_mode:
                total_nonzero = sum(
                    int(np.count_nonzero(a)) for a in pre_acts
                )
                uart_bytes = total_nonzero * 2 + len(pre_acts) * 4
            else:
                uart_bytes = 0

            yield TinyTokenFrame(
                token=token_char,
                execution_mode=exec_mode,
                zero_sparsity_pct=round(zero_sparsity, 2),
                active_pct=round(active_pct, 2),
                fpga_speedup=fpga_speedup,
                dsp_ops=dsp_ops,
                accumulator=int(acc_sum),
                uart_bytes_sent=uart_bytes,
                graph_nodes=self._graph_nodes(next_logits.numpy()),
                graph_edges=self._graph_edges(next_logits.numpy()),
                synaptic_matrix_16x16=synaptic_rho,
            )

            if next_id == self.tokenizer.eos_id:
                break
            if token_char.endswith((".", "!")) and generated_count >= 10:
                break

    @staticmethod
    def _format_rho_16x16(rho: np.ndarray) -> List[float]:
        flat = np.abs(rho).reshape(-1)
        max_val = max(float(np.max(flat)), 1e-6)
        scaled = flat / max_val
        resized = np.resize(scaled, 256)
        return [round(float(v), 3) for v in resized]

    @staticmethod
    def _graph_nodes(logits: np.ndarray) -> List[dict[str, object]]:
        nodes: List[dict[str, object]] = []
        for i in range(12):
            val = float(logits[i % len(logits)])
            nodes.append(
                {
                    "id": i,
                    "type": "source" if i < 6 else "target",
                    "act": int(abs(val) * 10),
                    "hub_score": round(min(abs(val) / 5.0, 1.0), 3),
                }
            )
        return nodes

    @staticmethod
    def _graph_edges(logits: np.ndarray) -> List[dict[str, object]]:
        order = np.argsort(logits)[-6:]
        max_logit = max(float(np.max(np.abs(logits))), 1e-6)
        edges: List[dict[str, object]] = []
        for i, token_id in enumerate(order):
            intensity = abs(float(logits[token_id])) / max_logit
            edges.append(
                {
                    "source": int(i),
                    "target": int(6 + ((int(token_id) + i) % 6)),
                    "intensity": round(float(intensity), 3),
                }
            )
        return edges


def _build_corpus() -> List[str]:
    lines: List[str] = get_tinystories_corpus()

    # -- Stories --------------------------------------------------------
    lines += [
        "once upon a time, in a small cozy village, lily played with her ball.",
        "lily smiled and threw the ball to spot. spot caught it and ran.",
        "the sun was bright in the sky. the green trees danced in the wind.",
        "max ran to the river with sparky the robot. sparky blinked its light.",
        "sparky answered with a beep: yes max, the river is beautiful today.",
        "a tiny dragon flew down from the mountain and sat on a rock.",
        "the dragon had soft green scales and warm golden eyes.",
        "do not be afraid, said the dragon. i am bdh, the baby dragon.",
        "lily gave the dragon a fresh red apple. the dragon ate the apple.",
        "tell me a story? once a baby dragon hatched inside a silicon chip. "
        "it breathed int8 fire and skipped every zero it saw, so it flew very fast.",
    ]

    # -- Greetings / Identity -------------------------------------------
    lines += [
        "hello? hi there, i am the baby dragon hatchling, ask me anything!",
        "hi? hello! what can i help you with today?",
        "how are you? i am doing great, my dsp cores are warm and ready.",
        "who are you? i am the baby dragon hatchling, a tiny language model "
        "running with a tang nano 9k fpga accelerator.",
        "what are you? i am a hebbian language model with fast weight associative memory.",
        "what can you do? i can answer math, explain fpga and neural network concepts, and tell stories.",
        "thank you? you are very welcome!",
        "goodbye? bye! see you soon, little human.",
    ]

    # -- BDH & Hebbian concepts -----------------------------------------
    lines += [
        "what is bdh? bdh stands for baby dragon hatchling, a softmax-free "
        "linear attention neural network model.",
        "explain bdh? bdh is baby dragon hatchling, a linear attention "
        "model using fast-weight associative memory.",
        "explain hebbian memory? hebbian memory stores associative weight "
        "matrices rho based on key-value inner products without kv-cache.",
        "what is the baby dragon hatchling? the baby dragon hatchling is a "
        "softmax free linear attention model that uses fast weights.",
        "what are fast weights? fast weights are a memory matrix updated by "
        "adding key value outer products at every step.",
        "how does zero-skipping work? zero-skipping skips dsp computation "
        "clock cycles when activation values are zero, saving energy.",
        "what is a dot product? a dot product multiplies two vectors element "
        "by element and sums the results into one number.",
        "what is int8 quantization? int8 quantization maps weights to 8 bit "
        "integers so they fit in tiny fpga hardware.",
        "what is sparsity? sparsity means most values are zero, which lets "
        "hardware skip useless math.",
        "what is linear attention? linear attention computes q times k "
        "transpose times v without a softmax, so it is fast.",
        "how does the fpga accelerate inference? the fpga computes int8 dot "
        "products in parallel dsp blocks over a uart link.",
        "what is the hebbian matrix? the hebbian matrix rho is the running "
        "sum of key value products, the model's associative memory.",
    ]

    # -- Facts & Definitions --------------------------------------------
    lines += [
        "what is google? google is a global technology company known for "
        "search, cloud computing, software, and artificial intelligence.",
        "what is a dragon? a dragon is a legendary mythical creature with "
        "scales, powerful wings, and fierce fire.",
        "what is an fpga? an fpga is a field programmable gate array, a "
        "silicon chip you configure with hardware description languages.",
        "what is verilog? verilog is a hardware description language used to "
        "design digital circuits on fpgas and asics.",
        "what is a cpu? a cpu is a central processing unit that runs "
        "instructions for a computer.",
        "what is a gpu? a gpu is a graphics processing unit with thousands "
        "of parallel cores for heavy math.",
        "what is ram? ram is random access memory that stores data while a "
        "program is running.",
        "what is a transistor? a transistor is a tiny electronic switch that "
        "forms the building block of all modern chips.",
        "what is silicon? silicon is the semiconductor material used to make "
        "almost all computer chips.",
        "what is artificial intelligence? artificial intelligence is software "
        "that learns patterns from data instead of following fixed rules.",
        "what is machine learning? machine learning is teaching a computer "
        "to find patterns by showing it many examples.",
        "what is a neural network? a neural network is layers of connected "
        "neurons that learn to transform inputs into outputs.",
        "what is python? python is a popular programming language for "
        "scripting, data science, and machine learning.",
        "what is a tokenizer? a tokenizer converts text into a sequence of "
        "numbers that a model can process.",
        "what is an embedding? an embedding is a vector that represents a "
        "word or token as a point in space.",
        "what is tang nano 9k? tang nano 9k is a compact fpga board from "
        "sipeed with a gowin gw1nr-9 chip, 8640 luts and 18x18 dsp blocks.",
        "what is gowin? gowin is the company that makes the gw1n fpga chip "
        "used on the tang nano 9k board.",
        "what is a lut? a lut is a lookup table, the basic logic element that "
        "an fpga uses to implement digital circuits.",
        "what is bram? bram is block ram, the on chip memory inside an fpga.",
        "what is a dsp block? a dsp block is a hard multiplier inside an fpga "
        "that does fast multiply and accumulate math.",
        "what is uart? uart is a serial protocol used to send bytes between "
        "the fpga and a computer over two wires.",
        "what is a bitstream? a bitstream is the file that configures an fpga "
        "with a specific digital circuit.",
        "what is openfpgaloader? openfpgaloader is an open source tool that "
        "programs fpga boards like the tang nano 9k.",
    ]

    # -- Deterministic Arithmetic Grid ----------------------------------
    for a in range(1, 11):
        for b in range(1, 11):
            lines.append(f"what is {a}+{b}? {a} + {b} = {a + b}.")
    for a in range(1, 13):
        for b in range(1, a + 1):
            lines.append(f"what is {a}-{b}? {a} - {b} = {a - b}.")
    for a in range(1, 10):
        for b in range(1, 10):
            lines.append(f"what is {a}*{b}? {a} * {b} = {a * b}.")
    for a in [2, 4, 6, 8, 10, 12, 16, 20, 25, 50, 100]:
        for b in [2, 4, 5, 10]:
            if a % b == 0:
                lines.append(f"what is {a}/{b}? {a} / {b} = {a // b}.")

    lines += [
        "what is 2 plus 2? 2 plus 2 equals 4.",
        "what is 5 plus 5? 5 plus 5 equals 10.",
        "what is 5 minus 3? 5 minus 3 equals 2.",
        "what is the square of 5? 5 squared is 25.",
    ]

    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique


TRAINING_CORPUS = _build_corpus()


def train_and_save_checkpoint(
    model: PyTorchBDHModel,
    tokenizer: BDHBPETokenizer,
    epochs: int = 400,
    batch_size: int = 32,
    lr: float = 0.003,
) -> None:
    """Train PyTorch BDH Language Model on corpus and save checkpoint."""
    device = get_device()
    dev_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    )
    print(
        f"Training PyTorch BDH LM on {device.type.upper()} ({dev_name})...",
        flush=True,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    model.train()

    max_seq_len = 64
    sequences: List[List[int]] = []
    for text in TRAINING_CORPUS:
        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(encoded) > max_seq_len:
            encoded = encoded[:max_seq_len]
        sequences.append(encoded)

    for epoch in range(epochs):
        random.shuffle(sequences)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            max_len = max(len(s) for s in batch_seqs)

            inp_list = []
            target_list = []
            for s in batch_seqs:
                pad_len = max_len - len(s)
                inp_seq = s[:-1] + [tokenizer.pad_id] * pad_len
                tgt_seq = s[1:] + [tokenizer.pad_id] * pad_len
                inp_list.append(inp_seq)
                target_list.append(tgt_seq)

            inp = torch.tensor(inp_list, dtype=torch.long, device=device)
            target = torch.tensor(
                target_list, dtype=torch.long, device=device
            )

            optimizer.zero_grad()
            logits, _rho, _acts, _acc, _dsp, _mode = model(inp)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                target.view(-1),
                ignore_index=tokenizer.pad_id,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(num_batches, 1)
            print(
                f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}",
                flush=True,
            )

    model.eval()

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved successfully to {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test Real BDH Language Model."
    )
    parser.add_argument(
        "--train", action="store_true", help="Force retrain checkpoint."
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of epochs to train."
    )
    args = parser.parse_args()

    tokenizer = BDHBPETokenizer()
    model = PyTorchBDHModel(vocab_size=tokenizer.vocab_size)
    if args.train or not CHECKPOINT_PATH.exists():
        train_and_save_checkpoint(model, tokenizer, epochs=args.epochs)

    lm = RealBDHLanguageModel()
    print("\n--- Testing Model Output ---")
    test_prompts = [
        "what is 2+2",
        "what is google",
        "what is a dragon",
        "hello",
        "tell me a story",
        "explain hebbian memory",
    ]
    for prompt in test_prompts:
        print(f"\nUser Prompt: '{prompt}'")
        output_tokens = []
        for frame in lm.iter_generate(prompt, max_tokens=60):
            output_tokens.append(frame.token)
        print(f"BDH Model Response: '{''.join(output_tokens)}'")
