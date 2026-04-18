"""
ZARX-1B Data Download Script
Downloads ProX dataset and ARC-AGI data from open sources.
Run this on Google Colab or Kaggle where you have disk space.
"""

import os
import json
import argparse
from pathlib import Path


def download_prox(output_dir: str, max_examples: int = 2000000, shard_size: int = 100000):
    """Download ProX refined code dataset from HuggingFace."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ProX dataset (streaming)...")
    try:
        ds = load_dataset("gair-prox/prox-code", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading gair-prox/prox-code: {e}")
        print("Trying alternative: gair-prox/prox-refined-web...")
        try:
            ds = load_dataset("gair-prox/prox-refined-web", split="train", streaming=True)
        except Exception as e2:
            print(f"Error: {e2}")
            print("Please check the HuggingFace collection: https://huggingface.co/collections/gair-prox")
            return

    shard = []
    shard_idx = 0
    total = 0

    for example in ds:
        shard.append(example)
        total += 1

        if len(shard) >= shard_size:
            path = output_dir / f"prox-shard-{shard_idx:04d}.jsonl"
            with open(path, "w") as f:
                for item in shard:
                    # Extract text field (may vary by dataset)
                    text = item.get("text", "") or item.get("content", "")
                    if text:
                        f.write(json.dumps({"text": text, "source": "prox"}) + "\n")

            print(f"  Saved shard {shard_idx}: {len(shard)} examples -> {path}")
            shard_idx += 1
            shard = []

        if total % 100000 == 0:
            print(f"  Downloaded {total} examples...")

        if total >= max_examples:
            break

    # Save remaining
    if shard:
        path = output_dir / f"prox-shard-{shard_idx:04d}.jsonl"
        with open(path, "w") as f:
            for item in shard:
                text = item.get("text", "") or item.get("content", "")
                if text:
                    f.write(json.dumps({"text": text, "source": "prox"}) + "\n")
        print(f"  Saved final shard {shard_idx}: {len(shard)} examples")

    print(f"\nProX download complete: {total} examples in {shard_idx + 1} shards")


def download_arc(output_dir: str):
    """Download ARC-AGI-1 and ARC-AGI-2 datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ARC-AGI-1
    arc1_dir = output_dir / "arc-agi-1"
    if not arc1_dir.exists():
        print("Cloning ARC-AGI-1...")
        os.system(f"git clone https://github.com/fchollet/ARC-AGI.git {arc1_dir}")
    else:
        print(f"ARC-AGI-1 already exists at {arc1_dir}")

    # ARC-AGI-2
    arc2_dir = output_dir / "arc-agi-2"
    if not arc2_dir.exists():
        print("Cloning ARC-AGI-2...")
        os.system(f"git clone https://github.com/arcprize/ARC-AGI-2.git {arc2_dir}")
    else:
        print(f"ARC-AGI-2 already exists at {arc2_dir}")

    # ReARC
    rearc_dir = output_dir / "re-arc"
    if not rearc_dir.exists():
        print("Cloning ReARC...")
        os.system(f"git clone https://github.com/michaelhodel/re-arc.git {rearc_dir}")
        print("Installing ReARC dependencies...")
        os.system(f"cd {rearc_dir} && pip install -r requirements.txt")
    else:
        print(f"ReARC already exists at {rearc_dir}")

    print("\nARC data download complete!")


def main():
    parser = argparse.ArgumentParser(description="ZARX-1B Data Downloader")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--download_prox", action="store_true", help="Download ProX dataset")
    parser.add_argument("--download_arc", action="store_true", help="Download ARC-AGI datasets")
    parser.add_argument("--download_all", action="store_true", help="Download everything")
    parser.add_argument("--max_prox_examples", type=int, default=2000000)
    args = parser.parse_args()

    if args.download_all:
        args.download_prox = True
        args.download_arc = True

    if not args.download_prox and not args.download_arc:
        print("Specify --download_prox, --download_arc, or --download_all")
        return

    if args.download_prox:
        download_prox(args.output_dir + "/prox", max_examples=args.max_prox_examples)

    if args.download_arc:
        download_arc(args.output_dir + "/arc")


if __name__ == "__main__":
    main()
