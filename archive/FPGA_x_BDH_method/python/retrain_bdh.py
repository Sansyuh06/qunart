"""Retrain BDH model with improved tokenizer and more epochs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bdh_bpe_tokenizer import BDHBPETokenizer
from bdh_real_llm import (
    PyTorchBDHModel,
    RealBDHLanguageModel,
    train_and_save_checkpoint,
)


def main() -> None:
    tokenizer = BDHBPETokenizer()
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}, "
          f"merges={len(tokenizer.merges)}")

    model = PyTorchBDHModel(vocab_size=tokenizer.vocab_size)
    train_and_save_checkpoint(
        model, tokenizer, epochs=800, batch_size=16, lr=0.003
    )

    print("\n=== Generation Quality Test ===")
    lm = RealBDHLanguageModel()
    prompts = [
        "what is 2+2",
        "what is a dragon",
        "hello",
        "tell me a story",
        "what is an fpga",
        "explain hebbian memory",
        "what is google",
        "who are you",
        "why is the sky blue",
        "what is 7+8",
    ]
    for p in prompts:
        tokens = [f.token for f in lm.iter_generate(p, max_tokens=80)]
        response = "".join(tokens)
        print(f"\nQ: {p}")
        print(f"A: {response}")


if __name__ == "__main__":
    main()
