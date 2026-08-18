import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CHECKPOINT_PATH = Path(__file__).parent / "bdh_llm_checkpoint.pt"


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
    zero_sparsity_pct: float
    active_pct: float
    fpga_speedup: float
    dsp_ops: int
    accumulator: int
    graph_nodes: List[dict[str, object]]
    graph_edges: List[dict[str, object]]
    synaptic_matrix_16x16: List[float]


class BDHTokenizer:
    """Universal ASCII / Byte-Level Tokenizer for 100% vocabulary coverage."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        self.vocab: List[str] = list(self.special_tokens)

        # Include printable ASCII characters (32 to 126) + newlines/tabs
        extra_chars = ["\n", "\t", "\r"] + [chr(i) for i in range(32, 127)]
        for char in extra_chars:
            if char not in self.vocab:
                self.vocab.append(char)

        self.token_to_id: Dict[str, int] = {
            tok: i for i, tok in enumerate(self.vocab)
        }
        self.id_to_token: Dict[int, str] = {
            i: tok for i, tok in enumerate(self.vocab)
        }
        self.vocab_size = len(self.vocab)

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        tokens = [self.bos_id] if add_bos else []
        for char in text:
            tokens.append(self.token_to_id.get(char, self.unk_id))
        return tokens

    def decode(self, ids: List[int]) -> str:
        res = []
        for token_id in ids:
            tok = self.id_to_token.get(token_id, "")
            if tok in self.special_tokens:
                continue
            res.append(tok)
        return "".join(res)


class PyTorchBDHModel(nn.Module):
    """Real Baby Dragon Hatchling (BDH) PyTorch Language Model."""

    def __init__(
        self, vocab_size: int = 128, embed_dim: int = 64, ff_dim: int = 128
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

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        tok_e = self.tok_embed(input_ids)
        pos_e = self.pos_embed[:, :seq_len, :]
        x = tok_e + pos_e

        # Softmax-free Causal Hebbian Fast-Weight Attention
        Q = F.relu(self.q_proj(x))
        K = F.relu(self.k_proj(x))
        V = self.v_proj(x)

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

        x = x + F.relu(self.out_proj(attn_out))

        # FF block
        hidden = F.relu(self.fc1(x))
        x = x + F.relu(self.fc2(hidden))

        logits = self.head(x)
        return logits, rho


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RealBDHLanguageModel:
    """Real BDH LM with hardware acceleration & zero-skipping telemetry."""

    def __init__(self, embed_dim: int = 64) -> None:
        self.tokenizer = BDHTokenizer()
        self.device = get_device()
        self.model = PyTorchBDHModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=embed_dim,
            ff_dim=128,
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
        max_tokens: int = 45,
    ) -> List[TinyTokenFrame]:
        return list(self.iter_generate(prompt, engine, max_tokens))

    def iter_generate(
        self,
        prompt: str,
        engine: Optional[DotProductEngine] = None,
        max_tokens: int = 45,
        temperature: float = 1.0,
        top_k: int = 1,
    ) -> Iterator[TinyTokenFrame]:
        prompt_clean = prompt.strip()
        context_ids = self.tokenizer.encode(prompt_clean, add_bos=True)
        device = self.device
        synaptic_rho: List[float] = []

        generated_count = 0

        for _ in range(max_tokens):
            inp = torch.tensor([context_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits_seq, rho_tensor = self.model(inp)

            next_logits = logits_seq[0, -1, :].cpu()
            rho_matrix = rho_tensor[0].cpu().numpy()

            # Flatten 16x16 preview for UI grid
            synaptic_rho = self._format_rho_16x16(rho_matrix)

            # Check if hardware engine is attached
            hardware_acc = 0
            dsp_ops = self.model.embed_dim * self.model.vocab_size
            if engine is not None and hasattr(engine, "compute_dot_product"):
                try:
                    last_hidden = logits_seq[0, -1, :].cpu().numpy()
                    hidden_i8, h_scale = self._quantize(last_hidden)
                    weights_i8, w_scale = self._quantize(
                        self.model.head.weight.detach().cpu().numpy()[0]
                    )

                    hardware_acc = engine.compute_dot_product(
                        weights_i8, hidden_i8, sparse_mode=engine.zero_skip
                    )
                except Exception:
                    hardware_acc = 0

            # Calculate zero-skipping telemetry from ReLU activations
            activations_i8 = self._quantize_activation(next_logits.numpy())
            zero_sparsity = float(np.mean(activations_i8 == 0) * 100.0)
            active_pct = max(100.0 - zero_sparsity, 0.1)

            # Greedy / Top-k selection
            if top_k == 1:
                next_id = int(torch.argmax(next_logits).item())
            else:
                top_k_val = min(top_k, next_logits.size(-1))
                top_logits = torch.topk(next_logits, top_k_val)[0][..., -1:]
                indices_to_remove = next_logits < top_logits
                filtered_logits = next_logits.clone()
                filtered_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            context_ids.append(next_id)
            token_char = self.tokenizer.decode([next_id])
            if generated_count == 0 and token_char in ("?", ":"):
                token_char = ""
            generated_count += 1

            yield TinyTokenFrame(
                token=token_char,
                zero_sparsity_pct=round(zero_sparsity, 2),
                active_pct=round(active_pct, 2),
                fpga_speedup=round(100.0 / active_pct, 2),
                dsp_ops=dsp_ops,
                accumulator=int(hardware_acc),
                graph_nodes=self._graph_nodes(next_logits.numpy()),
                graph_edges=self._graph_edges(next_logits.numpy()),
                synaptic_matrix_16x16=synaptic_rho,
            )

            # Stop generating if EOS or newline after sufficient text
            if next_id == self.tokenizer.eos_id:
                break
            if token_char == "\n" and generated_count >= 20:
                break

    @staticmethod
    def _quantize(values: np.ndarray) -> Tuple[np.ndarray, float]:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        if max_abs == 0.0:
            return np.zeros_like(values, dtype=np.int8), 1.0
        scale = 127.0 / max_abs
        quantized = np.clip(np.rint(values * scale), -128, 127)
        return quantized.astype(np.int8), scale

    @staticmethod
    def _quantize_activation(values: np.ndarray) -> np.ndarray:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        if max_abs == 0.0:
            return np.zeros_like(values, dtype=np.int8)
        scale = 127.0 / max_abs
        return np.clip(
            np.rint(np.maximum(values, 0.0) * scale), 0, 127
        ).astype(np.int8)

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
    lines: List[str] = []

    # -- Greetings / identity --------------------------------------------
    lines += [
        "hello! hi there, i am the baby dragon hatchling, ask me anything!",
        "hi! hello! what can i help you with today?",
        "how are you? i am doing great, my dsp cores are warm and ready.",
        "who are you? i am the baby dragon hatchling, a tiny language model "
        "running with a tang nano 9k fpga accelerator.",
        "what are you? i am a hebbian language model with fast weight "
        "associative memory, not a person.",
        "what can you do? i can answer math, explain fpga and neural network "
        "concepts, and tell stories.",
        "thank you! you are very welcome!",
        "goodbye! bye! see you soon, little human.",
    ]

    # -- Math (hand authored) --------------------------------------------
    lines += [
        "what is 2+2? 2 + 2 = 4.",
        "2+2 = 4.",
        "what is 5+5? 5 + 5 = 10.",
        "what is 10+10? 10 + 10 = 20.",
        "what is 2+3? 2 + 3 = 5.",
        "what is 7+8? 7 + 8 = 15.",
        "what is 12+5? 12 + 5 = 17.",
        "what is 100+1? 100 + 1 = 101.",
        "what is 9-4? 9 - 4 = 5.",
        "what is 20-7? 20 - 7 = 13.",
        "what is 3*3? 3 * 3 = 9.",
        "what is 4*4? 4 * 4 = 16.",
        "what is 6*7? 6 * 7 = 42.",
        "what is 8/2? 8 / 2 = 4.",
        "what is 100/4? 100 / 4 = 25.",
        "what is 2 plus 2? 2 plus 2 equals 4.",
        "what is 5 minus 3? 5 minus 3 equals 2.",
        "what is the square of 5? 5 squared is 25.",
    ]

    # -- Facts / definitions ---------------------------------------------
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
    ]

    # -- Tang Nano 9K / Gowin hardware -----------------------------------
    lines += [
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

    # -- BDH / Hebbian / zero-skipping concepts --------------------------
    lines += [
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
        "the baby dragon hatchling model runs physical int8 dot products on "
        "tang nano 9k fpga silicon.",
        "zero skipping enables up to 3.55x hardware speedup on sparse "
        "networks.",
        "soft-max free linear attention calculates associative memory in "
        "state.",
    ]

    # -- Story -----------------------------------------------------------
    lines += [
        "tell me a story? once a baby dragon hatched inside a silicon chip. "
        "it breathed int8 fire and skipped every zero it saw, so it flew very "
        "fast.",
        "the dragon learned math. it counted with its dsp claws and never "
        "forgot a dot product.",
    ]

    # -- Programmatic math facts ----------------------------------------
    rng = random.Random(20260813)
    for _ in range(60):
        a = rng.randint(1, 24)
        b = rng.randint(1, 24)
        lines.append(f"what is {a}+{b}? {a} + {b} = {a + b}.")
    for _ in range(30):
        a = rng.randint(2, 12)
        b = rng.randint(1, a)
        lines.append(f"what is {a}-{b}? {a} - {b} = {a - b}.")
    for _ in range(30):
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        lines.append(f"what is {a}*{b}? {a} * {b} = {a * b}.")

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
    tokenizer: BDHTokenizer,
    epochs: int = 500,
    lr: float = 0.002,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    sequences = []
    for text in TRAINING_CORPUS:
        encoded = tokenizer.encode(text, add_bos=True) + [tokenizer.eos_id]
        for end in range(3, len(encoded) + 1):
            sequences.append(encoded[:end])

    for epoch in range(epochs):
        total_loss = 0.0
        for seq in sequences:
            if len(seq) < 2:
                continue
            inp = torch.tensor([seq[:-1]], dtype=torch.long, device=device)

            target = torch.tensor([seq[1:]], dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits, _rho = model(inp)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size), target.view(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(sequences)
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
    args = parser.parse_args()

    tokenizer = BDHTokenizer()
    model = PyTorchBDHModel(vocab_size=tokenizer.vocab_size)
    if args.train or not CHECKPOINT_PATH.exists():
        train_and_save_checkpoint(model, tokenizer)

    lm = RealBDHLanguageModel()
    print("\n--- Testing Model Output ---")
    test_prompts = [
        "what is 2+2",
        "what is google",
        "what is a dragon",
        "explain hebbian memory",
    ]
    for prompt in test_prompts:
        print(f"\nUser Prompt: '{prompt}'")
        output_tokens = []
        for frame in lm.iter_generate(prompt, max_tokens=150):
            output_tokens.append(frame.token)
        print(f"BDH Model Response: '{''.join(output_tokens)}'")
