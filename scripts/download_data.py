"""
ZARX-1B Data Download Script
Downloads ProX dataset and ARC-AGI data from open sources.

ProX Datasets Available on HuggingFace:
- gair-prox/FineWeb-pro     → 100B tokens (general, refined from FineWeb)
- gair-prox/c4-pro          → Refined from C4
- gair-prox/RedPajama-pro   → Refined from RedPajama (includes code!)
- gair-prox/open-web-math-pro → 5B tokens (math)

For ZARX-1B we use:
- RedPajama-pro (contains code from StarCoder/RedPajama)
- FineWeb-pro (general high-quality text including code snippets)
- ARC-AGI-1 + ARC-AGI-2 + ReARC (reasoning data)
"""

import os
import json
import argparse
from pathlib import Path


def download_prox(output_dir: str, dataset: str = "RedPajama-pro",
                   max_examples: int = 2000000, shard_size: int = 100000):
    """Download ProX refined dataset from HuggingFace.

    Available datasets:
    - 'RedPajama-pro' (recommended - contains code from StarCoder data)
    - 'FineWeb-pro' (100B tokens, general web text including code)
    - 'c4-pro' (refined C4)
    - 'open-web-math-pro' (5B math tokens)
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_id = f"gair-prox/{dataset}"
    print(f"Downloading {hf_id} (streaming)...")
    print(f"  Max examples: {max_examples:,}")
    print(f"  Output: {output_dir}")

    try:
        ds = load_dataset(hf_id, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading {hf_id}: {e}")
        print(f"Trying with 'default' config...")
        try:
            ds = load_dataset(hf_id, split="train", streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"Failed: {e2}")
            print("Available datasets: RedPajama-pro, FineWeb-pro, c4-pro, open-web-math-pro")
            return

    shard = []
    shard_idx = 0
    total = 0
    kept = 0

    for example in ds:
        total += 1

        # Extract text (field name varies by dataset)
        text = example.get("text", "") or example.get("content", "") or example.get("document", "")
        if not text or len(text) < 50:
            continue

        # For RedPajama-pro: check if it's code-related
        # RedPajama includes: github, stackexchange, arxiv, books, etc.
        meta = example.get("meta", {}) or example.get("metadata", {})
        source_tag = meta.get("redpajama_source", "") or meta.get("source", "")

        # Tag the source
        source = "prox"
        if source_tag:
            source = f"prox_{source_tag}"

        shard.append({"text": text, "source": source})

        if len(shard) >= shard_size:
            path = output_dir / f"prox-shard-{shard_idx:04d}.jsonl"
            with open(path, "w") as f:
                for item in shard:
                    f.write(json.dumps(item) + "\n")
            print(f"  Saved shard {shard_idx}: {len(shard)} examples -> {path.name}")
            shard_idx += 1
            shard = []

        kept += 1
        if kept % 100000 == 0:
            print(f"  Progress: {kept:,} kept / {total:,} scanned")

        if kept >= max_examples:
            break

    # Save remaining
    if shard:
        path = output_dir / f"prox-shard-{shard_idx:04d}.jsonl"
        with open(path, "w") as f:
            for item in shard:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved final shard {shard_idx}: {len(shard)} examples")

    print(f"\nDownload complete: {kept:,} examples in {shard_idx + 1} shards")
    print(f"  Scanned: {total:,} total, Kept: {kept:,}")


def download_prox_code_only(output_dir: str, max_examples: int = 1000000):
    """Download ProX data filtered for code content.

    Strategy: Download RedPajama-pro and filter for github/code sources.
    RedPajama's GitHub subset contains high-quality code.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading RedPajama-pro (code-filtered)...")
    try:
        ds = load_dataset("gair-prox/RedPajama-pro", split="train", streaming=True)
    except Exception as e:
        print(f"Error: {e}")
        print("Trying FineWeb-pro as fallback...")
        return download_prox(output_dir, dataset="FineWeb-pro", max_examples=max_examples)

    count = 0
    shard = []
    shard_idx = 0

    for example in ds:
        text = example.get("text", "")
        if not text or len(text) < 50:
            continue

        # Simple code detection heuristic
        code_indicators = [
            "def ", "class ", "import ", "from ", "function ",
            "return ", "if ", "for ", "while ", "#include",
            "pub fn ", "fn main", "func ", "package ",
            "console.log", "=> ", "-> ", "module ",
            "```", "    ", "\t",
        ]
        code_score = sum(1 for ind in code_indicators if ind in text)

        # Keep if it looks like code (score >= 2)
        if code_score < 2:
            continue

        shard.append({"text": text, "source": "prox_code"})

        if len(shard) >= 100000:
            path = output_dir / f"prox-code-shard-{shard_idx:04d}.jsonl"
            with open(path, "w") as f:
                for item in shard:
                    f.write(json.dumps(item) + "\n")
            print(f"  Saved code shard {shard_idx}: {len(shard)} examples")
            shard_idx += 1
            shard = []

        count += 1
        if count % 100000 == 0:
            print(f"  Code examples found: {count:,}")

        if count >= max_examples:
            break

    if shard:
        path = output_dir / f"prox-code-shard-{shard_idx:04d}.jsonl"
        with open(path, "w") as f:
            for item in shard:
                f.write(json.dumps(item) + "\n")

    print(f"\nCode-filtered download: {count:,} code examples")


def download_arc(output_dir: str):
    """Download ARC-AGI-1, ARC-AGI-2, and ReARC datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ARC-AGI-1
    arc1_dir = output_dir / "arc-agi-1"
    if not (arc1_dir / "data").exists():
        print("Cloning ARC-AGI-1...")
        os.system(f"git clone https://github.com/fchollet/ARC-AGI.git {arc1_dir}")
    else:
        print(f"ARC-AGI-1 already exists at {arc1_dir}")

    # ARC-AGI-2
    arc2_dir = output_dir / "arc-agi-2"
    if not (arc2_dir / "data").exists():
        print("Cloning ARC-AGI-2...")
        os.system(f"git clone https://github.com/arcprize/ARC-AGI-2.git {arc2_dir}")
    else:
        print(f"ARC-AGI-2 already exists at {arc2_dir}")

    # ReARC (procedural generators)
    rearc_dir = output_dir / "re-arc"
    if not rearc_dir.exists():
        print("Cloning ReARC...")
        os.system(f"git clone https://github.com/michaelhodel/re-arc.git {rearc_dir}")
        print("Installing ReARC dependencies...")
        os.system(f"cd {rearc_dir} && pip install -r requirements.txt 2>/dev/null")
    else:
        print(f"ReARC already exists at {rearc_dir}")

    # Generate ReARC data
    rearc_data = output_dir / "rearc_generated"
    if not rearc_data.exists():
        print("\nGenerating ReARC data (1000 examples per task)...")
        os.makedirs(rearc_data, exist_ok=True)

        try:
            sys_path = str(rearc_dir)
            import sys
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)

            from main import generate_dataset
            generate_dataset(
                path=str(rearc_data),
                seed=42,
                n_examples=1000,  # 1000 per task
                diff_lb=0,
                diff_ub=1,
            )
            print(f"ReARC data generated at {rearc_data}")
        except Exception as e:
            print(f"ReARC generation failed: {e}")
            print("You can generate it later by running:")
            print(f"  cd {rearc_dir} && python -c 'from main import generate_dataset; generate_dataset(\"{rearc_data}\", seed=42, n_examples=1000)'")

    print("\nARC data download complete!")
    print(f"  ARC-AGI-1: {arc1_dir}/data/training/ (400 tasks)")
    print(f"  ARC-AGI-2: {arc2_dir}/data/training/ (1000 tasks)")
    print(f"  ReARC: {rearc_data}/ (400 tasks × 1000 examples)")


def main():
    parser = argparse.ArgumentParser(description="ZARX-1B Data Downloader")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--download_prox", action="store_true", help="Download ProX dataset")
    parser.add_argument("--download_prox_code", action="store_true", help="Download ProX code-filtered data")
    parser.add_argument("--download_arc", action="store_true", help="Download ARC-AGI datasets")
    parser.add_argument("--download_all", action="store_true", help="Download everything")
    parser.add_argument("--prox_dataset", type=str, default="RedPajama-pro",
                       choices=["RedPajama-pro", "FineWeb-pro", "c4-pro", "open-web-math-pro"],
                       help="Which ProX dataset to download")
    parser.add_argument("--max_prox_examples", type=int, default=2000000)
    args = parser.parse_args()

    if args.download_all:
        args.download_prox = True
        args.download_arc = True

    if not args.download_prox and not args.download_arc and not args.download_prox_code:
        print("Specify --download_prox, --download_prox_code, --download_arc, or --download_all")
        return

    if args.download_prox_code:
        download_prox_code_only(args.output_dir + "/prox-code", max_examples=args.max_prox_examples)
    elif args.download_prox:
        download_prox(args.output_dir + "/prox", dataset=args.prox_dataset, max_examples=args.max_prox_examples)

    if args.download_arc:
        download_arc(args.output_dir + "/arc")


if __name__ == "__main__":
    main()
