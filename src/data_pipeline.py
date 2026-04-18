"""
ZARX-1B Data Pipeline
Handles data loading, preprocessing, deduplication, and format unification.
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from collections import defaultdict

from datasketch import MinHash, MinHashLSH


class DataPipeline:
    """Unified data pipeline for ZARX-1B pretraining data."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        target_languages: List[str] = None,
        language_weights: Dict[str, float] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.target_languages = target_languages or [
            "python", "javascript", "typescript", "rust",
            "cpp", "java", "go",
        ]

        self.language_weights = language_weights or {
            "python": 0.40,
            "javascript": 0.15,
            "typescript": 0.10,
            "cpp": 0.10,
            "java": 0.10,
            "rust": 0.08,
            "go": 0.07,
        }

    def detect_language(self, text: str, filename: str = "") -> str:
        """Detect programming language from file content and extension."""
        ext_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rs": "rust",
            ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
            ".java": "java",
            ".go": "go",
        }

        if filename:
            ext = Path(filename).suffix.lower()
            if ext in ext_lang:
                return ext_lang[ext]

        # Fallback: heuristic detection
        if "def " in text and "import " in text:
            return "python"
        if "fn " in text and "let " in text and "::" in text:
            return "rust"
        if "func " in text and "package " in text:
            return "go"
        if "public class" in text or "private void" in text:
            return "java"
        if "console.log" in text or "=> " in text:
            return "javascript"
        if "#include" in text and "int main" in text:
            return "cpp"

        return "unknown"

    def quality_filter(self, text: str, min_lines: int = 5, max_lines: int = 500) -> bool:
        """Filter low-quality code."""
        lines = text.split("\n")
        num_lines = len(lines)

        if num_lines < min_lines or num_lines > max_lines:
            return False

        # Reject auto-generated code
        auto_gen_patterns = [
            r"auto-generated",
            r"automatically generated",
            r"do not edit",
            r"this file was generated",
            r"codegen",
        ]
        text_lower = text.lower()
        for pattern in auto_gen_patterns:
            if re.search(pattern, text_lower):
                return False

        # Reject files with >50% blank lines
        blank_lines = sum(1 for line in lines if not line.strip())
        if blank_lines / num_lines > 0.5:
            return False

        # Reject files with very low info density (mostly repeated lines)
        unique_lines = len(set(line.strip() for line in lines if line.strip()))
        if unique_lines < 3:
            return False

        return True

    def deduplicate_minhash(
        self,
        examples: List[Dict],
        threshold: float = 0.7,
        num_perm: int = 128,
    ) -> List[Dict]:
        """Remove near-duplicate documents using MinHash LSH."""
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        unique_examples = []

        for i, example in enumerate(examples):
            text = example.get("text", "")
            if not text:
                continue

            # Create MinHash
            tokens = text.lower().split()
            mh = MinHash(num_perm=num_perm)
            for token in tokens[:500]:  # Use first 500 tokens for speed
                mh.update(token.encode("utf8"))

            # Check if duplicate
            key = f"doc_{i}"
            if lsh.query(mh):
                continue  # Duplicate found

            lsh.insert(key, mh)
            unique_examples.append(example)

        removed = len(examples) - len(unique_examples)
        print(f"Deduplication: {len(examples)} -> {len(unique_examples)} (removed {removed} duplicates)")
        return unique_examples

    def process_jsonl_file(
        self,
        input_path: str,
        output_path: str,
        source: str = "prox",
        apply_filters: bool = True,
        apply_dedup: bool = True,
        max_examples: Optional[int] = None,
    ) -> Dict:
        """Process a single JSONL file."""
        examples = []
        stats = {"total": 0, "filtered": 0, "kept": 0, "by_lang": defaultdict(int)}

        with open(input_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                stats["total"] += 1

                try:
                    example = json.loads(line)
                    text = example.get("text", "") or example.get("content", "")
                    if not text:
                        stats["filtered"] += 1
                        continue

                    # Quality filter
                    if apply_filters and not self.quality_filter(text):
                        stats["filtered"] += 1
                        continue

                    # Detect language
                    lang = self.detect_language(text, example.get("filename", ""))
                    if lang not in self.target_languages and lang != "unknown":
                        stats["filtered"] += 1
                        continue

                    stats["by_lang"][lang] += 1
                    stats["kept"] += 1

                    examples.append({
                        "text": text,
                        "source": source,
                        "language": lang,
                    })

                    if max_examples and stats["kept"] >= max_examples:
                        break

                except json.JSONDecodeError:
                    stats["filtered"] += 1
                    continue

        # Deduplicate
        if apply_dedup and len(examples) > 100:
            examples = self.deduplicate_minhash(examples)

        # Write output
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        print(f"Processed {input_path}: {stats}")
        return stats

    def merge_and_weight(
        self,
        input_dir: str,
        output_path: str,
        mix_ratio: Dict[str, float] = None,
    ) -> None:
        """Merge processed files with weighted sampling."""
        mix_ratio = mix_ratio or {"prox": 0.80, "arc": 0.20}

        # Load examples by source
        by_source = defaultdict(list)
        input_path = Path(input_dir)

        for f in input_path.glob("*.jsonl"):
            source = f.stem.split("_")[0]  # e.g., "prox_shard_0000" -> "prox"
            with open(f) as fp:
                for line in fp:
                    if line.strip():
                        example = json.loads(line)
                        example_source = example.get("source", source)
                        by_source[example_source].append(example)

        # Sample according to mix ratio
        total_target = sum(len(v) for v in by_source.values())
        sampled = []

        for source, ratio in mix_ratio.items():
            if source not in by_source:
                print(f"Warning: No data for source '{source}'")
                continue

            n_samples = int(total_target * ratio)
            examples = by_source[source]
            if len(examples) > n_samples:
                import random
                examples = random.sample(examples, n_samples)
            sampled.extend(examples)
            print(f"  {source}: {len(examples)} examples (ratio: {ratio})")

        # Shuffle
        import random
        random.shuffle(sampled)

        # Write
        with open(output_path, "w") as f:
            for example in sampled:
                f.write(json.dumps(example) + "\n")

        print(f"Merged dataset: {len(sampled)} examples -> {output_path}")


def main():
    """Run the data pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="ZARX-1B Data Pipeline")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mode", type=str, default="process", choices=["process", "merge"])
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--source", type=str, default="prox")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    pipeline = DataPipeline(raw_dir=args.raw_dir, processed_dir=args.processed_dir)

    if args.mode == "process" and args.input_file:
        output = args.output_file or str(Path(args.processed_dir) / f"{args.source}_processed.jsonl")
        pipeline.process_jsonl_file(
            input_path=args.input_file,
            output_path=output,
            source=args.source,
            max_examples=args.max_examples,
        )
    elif args.mode == "merge":
        output = args.output_file or str(Path(args.processed_dir) / "zarx_pretrain.jsonl")
        pipeline.merge_and_weight(
            input_dir=args.processed_dir,
            output_path=output,
        )


if __name__ == "__main__":
    main()
