"""Byte-Level Byte-Pair Encoding (BPE) Tokenizer for BDH Language Models."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_VOCAB_PATH = Path(__file__).parent / "bdh_bpe_vocab.json"


def get_tinystories_corpus() -> List[str]:
    """Return a rich TinyStories corpus for tokenizer and LM training."""
    stories = [
        "once upon a time, there was a little girl named lily.",
        "she loved to play in the garden with her red ball.",
        "one sunny day, lily saw a tiny dragon sitting on a green leaf.",
        "the dragon was small and shiny with soft green scales.",
        "the dragon breathed a tiny puff of warm fire into the air.",
        "lily smiled brightly and gave the dragon a sweet red apple.",
        "the dragon munched happily on the apple and flapped its wings.",
        "tim and his brave dog spot went for a walk in the dark forest.",
        "spot saw a funny fluffy rabbit near the big oak tree.",
        "spot barked happily and ran after the rabbit in the grass.",
        "the rabbit hopped quickly into a small hole under the roots.",
        "tim called spot back. tim gave spot a tasty bone.",
        "max had a clever little robot named sparky.",
        "sparky had bright blue lights and a warm metal heart.",
        "sparky could count numbers very fast: 1, 2, 3, 4, 5!",
        "max laughed and clapped his hands in joy as sparky danced.",
        "once upon a time, a baby dragon hatched inside a silicon chip.",
        "the dragon was named bdh. it lived inside an fpga board.",
        "bdh loved to calculate dot products on the dsp cores.",
        "it skipped every zero activation to run super fast and cool.",
        "the baby dragon learned math. what is 2+2? 2 + 2 = 4.",
        "what is 5+5? 5 + 5 = 10. what is 10+10? 10 + 10 = 20.",
        "zero-skipping saves clock cycles when activations are zero.",
        "the hebbian memory matrix stores key value associations.",
        "softmax-free linear attention computes state transitions fast.",
        "lily, tim, and max built a cozy wooden house near the river.",
        "they shared stories, sang songs, and lived happily ever after.",
    ]
    return stories


class BDHBPETokenizer:
    """Byte-Level BPE Tokenizer with chunk pre-tokenization."""

    def __init__(
        self,
        vocab_file: Optional[Path] = None,
        target_vocab_size: int = 384,
        corpus: Optional[List[str]] = None,
    ) -> None:
        if not (256 <= target_vocab_size <= 512):
            raise ValueError("target_vocab_size must be between 256 and 512")

        self.vocab_file = vocab_file or DEFAULT_VOCAB_PATH
        self.target_vocab_size = target_vocab_size

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

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.id_to_bytes: Dict[int, bytes] = {}
        for b in range(256):
            self.id_to_bytes[4 + b] = bytes([b])

        self.merges: Dict[Tuple[int, int], int] = {}
        self.ranks: Dict[Tuple[int, int], int] = {}

        self.id_to_token: Dict[int, str] = {
            0: self.pad_token,
            1: self.bos_token,
            2: self.eos_token,
            3: self.unk_token,
        }
        for b in range(256):
            tok_str = bytes([b]).decode("utf-8", errors="replace")
            self.id_to_token[4 + b] = tok_str

        if self.vocab_file.exists():
            self.load(self.vocab_file)
        else:
            training_corpus = corpus or get_tinystories_corpus()
            self.train(training_corpus, target_vocab_size=target_vocab_size)
            self.save(self.vocab_file)

        self._rebuild_token_mappings()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes) + 4

    def _rebuild_token_mappings(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        for idx, tok in self.id_to_token.items():
            self.token_to_id[tok] = idx

    @staticmethod
    def _chunk(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+|[0-9]+|\s+|[^\s\w]", text)

    def train(
        self, corpus: List[str], target_vocab_size: int = 384
    ) -> None:
        """Train Byte-Level BPE merges on word chunks within a corpus."""
        if not (256 <= target_vocab_size <= 512):
            raise ValueError("target_vocab_size must be between 256 and 512")
        self.target_vocab_size = target_vocab_size

        self.id_to_bytes = {}
        for b in range(256):
            self.id_to_bytes[4 + b] = bytes([b])
        self.merges = {}
        self.ranks = {}

        self.id_to_token = {
            0: self.pad_token,
            1: self.bos_token,
            2: self.eos_token,
            3: self.unk_token,
        }
        for b in range(256):
            tok_str = bytes([b]).decode("utf-8", errors="replace")
            self.id_to_token[4 + b] = tok_str

        # Chunk sequences to avoid merging words with numbers or punctuation
        sequences: List[List[int]] = []
        for doc in corpus:
            chunks = self._chunk(doc)
            for ch in chunks:
                if len(ch) > 1:
                    b_seq = list(ch.encode("utf-8"))
                    sequences.append([b + 4 for b in b_seq])

        num_merges = target_vocab_size - 260
        rank = 0

        for _ in range(num_merges):
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            if pair_counts[best_pair] < 2:
                break

            new_id = len(self.id_to_bytes) + 4
            self.merges[best_pair] = new_id
            self.ranks[best_pair] = rank
            rank += 1

            merged_bytes = (
                self.id_to_bytes[best_pair[0]]
                + self.id_to_bytes[best_pair[1]]
            )
            self.id_to_bytes[new_id] = merged_bytes
            self.id_to_token[new_id] = merged_bytes.decode(
                "utf-8", errors="replace"
            )

            new_sequences: List[List[int]] = []
            for seq in sequences:
                new_seq: List[int] = []
                i = 0
                while i < len(seq):
                    if (
                        i < len(seq) - 1
                        and (seq[i], seq[i + 1]) == best_pair
                    ):
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

        self._rebuild_token_mappings()

    def _encode_chunk(self, chunk: str) -> List[int]:
        raw_bytes = chunk.encode("utf-8")
        tokens = [b + 4 for b in raw_bytes]

        while len(tokens) >= 2:
            pairs = [
                (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)
            ]
            mergeable_pairs = [p for p in pairs if p in self.ranks]
            if not mergeable_pairs:
                break

            best_pair = min(mergeable_pairs, key=lambda p: self.ranks[p])
            new_id = self.merges[best_pair]

            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = False
    ) -> List[int]:
        """Encode text to token IDs using chunk-level BPE merges."""
        chunks = self._chunk(text)
        tokens: List[int] = []
        for ch in chunks:
            tokens.extend(self._encode_chunk(ch))

        res = []
        if add_bos:
            res.append(self.bos_id)
        res.extend(tokens)
        if add_eos:
            res.append(self.eos_id)
        return res

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a string."""
        byte_chunks: List[bytes] = []
        for token_id in ids:
            if token_id in (
                self.pad_id, self.bos_id, self.eos_id, self.unk_id
            ):
                continue
            if token_id in self.id_to_bytes:
                byte_chunks.append(self.id_to_bytes[token_id])
        raw_data = b"".join(byte_chunks)
        return raw_data.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        """Save vocabulary and BPE merge rules to JSON."""
        merges_serialized = [
            {"pair": list(pair), "new_id": new_id, "rank": self.ranks[pair]}
            for pair, new_id in self.merges.items()
        ]
        data = {
            "target_vocab_size": self.target_vocab_size,
            "merges": merges_serialized,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load vocabulary and BPE merge rules from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.target_vocab_size = data.get("target_vocab_size", 384)
        self.merges = {}
        self.ranks = {}

        self.id_to_bytes = {}
        for b in range(256):
            self.id_to_bytes[4 + b] = bytes([b])

        self.id_to_token = {
            0: self.pad_token,
            1: self.bos_token,
            2: self.eos_token,
            3: self.unk_token,
        }
        for b in range(256):
            tok_str = bytes([b]).decode("utf-8", errors="replace")
            self.id_to_token[4 + b] = tok_str

        for item in data.get("merges", []):
            pair = (int(item["pair"][0]), int(item["pair"][1]))
            new_id = int(item["new_id"])
            rank = int(item["rank"])
            self.merges[pair] = new_id
            self.ranks[pair] = rank

            merged_bytes = (
                self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
            )
            self.id_to_bytes[new_id] = merged_bytes
            self.id_to_token[new_id] = merged_bytes.decode(
                "utf-8", errors="replace"
            )

        self._rebuild_token_mappings()
