"""Fine-tune the BDH checkpoint to clean up mid-answer drift.

This is the VERIFIED-safe fine-tune: a gentle low-LR continuation of the
original checkpoint on the ORIGINAL question corpus (all-prefixes). It removes
the mid-answer garble (e.g. "searck cyaloerful" -> "search, cloud computing")
without collapsing the model.

WARNING: do NOT retrain with a much larger corpus at this model size
(64-dim). Attempting to add many new topics at once causes catastrophic
forgetting — the model reallocates its limited capacity and loses the answers
it already memorized. To add real new knowledge, either:
  * use a bigger model (embed_dim 96-128) + a GPU / several hours of CPU, or
  * add new lines a few at a time with interleaved fine-tuning.

Usage:
    python python/finetune_bdh.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from bdh_real_llm import (  # noqa: E402
    BDHTokenizer,
    PyTorchBDHModel,
    RealBDHLanguageModel,
    CHECKPOINT_PATH,
)

# The original corpus the checkpoint was trained on (keep this fixed).
CORPUS = [
    "what is 2+2? 2 + 2 = 4.",
    "2+2 = 4.",
    "what is 5+5? 5 + 5 = 10.",
    "what is 10+10? 10 + 10 = 20.",
    "what is google? google is a global technology company known for "
    "search, cloud computing, software, and artificial intelligence.",
    "what is a dragon? a dragon is a legendary mythical creature with "
    "scales, powerful wings, and fierce fire.",
    "what is an fpga? an fpga is a field-programmable gate array silicon "
    "chip configured using hardware description languages like verilog.",
    "what is tang nano 9k? tang nano 9k is a compact fpga development "
    "board featuring a gowin gw1nr-9 chip with 8640 luts and 18x18 dsp.",
    "explain hebbian memory? hebbian memory stores associative weight "
    "matrices rho based on key-value inner products without kv-cache.",
    "how does zero-skipping work? zero-skipping skips dsp computation "
    "clock cycles when activation values are zero, saving energy.",
    "the baby dragon hatchling model runs physical int8 dot products on "
    "tang nano 9k fpga silicon.",
    "zero skipping enables up to 3.55x hardware speedup on sparse networks.",
    "soft-max free linear attention calculates associative memory in state.",
]


def main() -> None:
    epochs = 15
    lr = 0.0002

    tokenizer = BDHTokenizer()
    model = PyTorchBDHModel(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu",
                                     weights_only=True))

    sequences = []
    for text in CORPUS:
        encoded = tokenizer.encode(text, add_bos=True) + [tokenizer.eos_id]
        for end in range(3, len(encoded) + 1):
            sequences.append(encoded[:end])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    print(f"Fine-tuning on {len(CORPUS)} lines "
          f"({len(sequences)} steps/epoch), {epochs} epochs, lr={lr}",
          flush=True)
    for epoch in range(epochs):
        total = 0.0
        for seq in sequences:
            inp = torch.tensor([seq[:-1]])
            tgt = torch.tensor([seq[1:]])
            optimizer.zero_grad()
            logits, _ = model(inp)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size), tgt.view(-1)
            )
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: "
                  f"{total / len(sequences):.4f}", flush=True)

    model.eval()
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved fine-tuned checkpoint to {CHECKPOINT_PATH}", flush=True)

    lm = RealBDHLanguageModel()
    print("\n--- Fine-tuned output ---")
    for prompt in ["what is 2+2", "what is google", "what is a dragon",
                   "explain hebbian memory"]:
        toks = [f.token for f in lm.iter_generate(prompt, max_tokens=90)]
        print(f"'{prompt}' -> '{''.join(toks)}'")


if __name__ == "__main__":
    main()
