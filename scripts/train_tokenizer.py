"""
ZARX-1B Tokenizer Training Script
Trains a custom BPE tokenizer on the code + ARC corpus.
"""

import json
import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


def train_tokenizer(
    data_path: str = "data/processed",
    output_path: str = "configs/tokenizer.json",
    vocab_size: int = 32000,
    min_frequency: int = 50,
    max_lines: int = 5_000_000,
):
    """Train a BPE tokenizer on the ZARX-1B corpus."""

    print(f"Training ZARX-1B tokenizer...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Data path: {data_path}")
    print(f"  Max lines: {max_lines:,}")

    # Collect training text
    def text_iterator():
        count = 0
        data_dir = Path(data_path)
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            print(f"  Reading: {jsonl_file}")
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        example = json.loads(line)
                        text = example.get("text", "")
                        if text:
                            yield text
                            count += 1
                            if count >= max_lines:
                                print(f"  Reached {max_lines:,} lines, stopping.")
                                return
                    except json.JSONDecodeError:
                        continue

    # Special tokens
    special_tokens = [
        "<pad>", "<eos>", "<bos>", "<unk>",
        "<code>", "</code>",
        "<arc_input>", "</arc_input>",
        "<arc_output>", "</arc_output>",
        "<solution>", "</solution>",
    ]

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Add padding
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

    # Save
    tokenizer.save(output_path)
    print(f"\nTokenizer saved to: {output_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Test
    test_strings = [
        "def hello_world():\n    print('Hello!')",
        "for i in range(10):\n    x += i",
        "class Solution:\n    def solve(self, grid):",
        "import numpy as np\narr = np.zeros((3, 3))",
    ]

    print("\nTokenizer tests:")
    for s in test_strings:
        output = tokenizer.encode(s)
        ratio = len(s) / len(output.ids) if len(output.ids) > 0 else 0
        print(f"  '{s[:50]}...' -> {len(output.ids)} tokens (ratio: {ratio:.1f} chars/token)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZARX-1B Tokenizer Training")
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument("--output_path", type=str, default="configs/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=50)
    parser.add_argument("--max_lines", type=int, default=5_000_000)
    args = parser.parse_args()

    train_tokenizer(
        data_path=args.data_path,
        output_path=args.output_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_lines=args.max_lines,
    )
