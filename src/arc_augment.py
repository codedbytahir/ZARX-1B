"""
ZARX-1B ARC Augmentation Pipeline
Generates augmented ARC-AGI tasks for training.

Sources:
- ARC-AGI-1: 400 training tasks
- ARC-AGI-2: 1000 training tasks
- ReARC: Unlimited procedural augmentations

Augmentation strategies:
- Color permutation
- Rotation (90, 180, 270)
- Horizontal/vertical flip
- Grid scaling (2x, 3x)
- Noise injection
- Object translation
"""

import json
import random
import copy
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import permutations

import numpy as np


class ARCAugmenter:
    """Generates augmented ARC tasks from original tasks."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def load_tasks(self, path: str) -> Dict:
        """Load ARC tasks from JSON directory."""
        tasks = {}
        task_dir = Path(path)

        if task_dir.is_file() and task_dir.suffix == ".json":
            with open(task_dir) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    tasks.update(data)
                elif isinstance(data, list):
                    for i, task in enumerate(data):
                        tasks[f"task_{i}"] = task
        else:
            for f in sorted(task_dir.glob("*.json")):
                with open(f) as fp:
                    try:
                        data = json.load(fp)
                        if isinstance(data, dict) and "train" in data:
                            tasks[f.stem] = data
                        elif isinstance(data, dict):
                            # Might be a collection of tasks
                            for k, v in data.items():
                                if isinstance(v, dict) and "train" in v:
                                    tasks[k] = v
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(tasks)} ARC tasks from {path}")
        return tasks

    # ==================== AUGMENTATION METHODS ====================

    def permute_colors(self, grid: List[List[int]], mapping: Optional[Dict] = None) -> List[List[int]]:
        """Remap colors using a permutation of 0-9."""
        if mapping is None:
            colors = list(range(10))
            self.rng.shuffle(colors)
            mapping = {i: colors[i] for i in range(10)}
        return [[mapping[cell] for cell in row] for row in grid]

    def rotate_90(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise."""
        return [list(row) for row in zip(*grid[::-1])]

    def rotate_180(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 180 degrees."""
        return self.rotate_90(self.rotate_90(grid))

    def rotate_270(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 270 degrees clockwise."""
        return self.rotate_90(self.rotate_180(grid))

    def flip_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid horizontally."""
        return [row[::-1] for row in grid]

    def flip_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid vertically."""
        return grid[::-1]

    def scale_grid(self, grid: List[List[int]], factor: int = 2) -> List[List[int]]:
        """Scale grid by integer factor."""
        new_grid = []
        for row in grid:
            new_row = []
            for cell in row:
                new_row.extend([cell] * factor)
            for _ in range(factor):
                new_grid.append(new_row[:])
        return new_grid

    def add_noise(self, grid: List[List[int]], num_pixels: int = 2) -> List[List[int]]:
        """Add random noise pixels to grid."""
        grid = copy.deepcopy(grid)
        h, w = len(grid), len(grid[0])
        for _ in range(num_pixels):
            i, j = self.rng.randint(0, h - 1), self.rng.randint(0, w - 1)
            original = grid[i][j]
            new_color = self.rng.choice([c for c in range(10) if c != original])
            grid[i][j] = new_color
        return grid

    def translate_grid(self, grid: List[List[int]], dx: int = 0, dy: int = 0) -> List[List[int]]:
        """Shift grid contents by (dx, dy), with zero-fill."""
        h, w = len(grid), len(grid[0])
        new_grid = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                ni, nj = i + dy, j + dx
                if 0 <= ni < h and 0 <= nj < w:
                    new_grid[ni][nj] = grid[i][j]
        return new_grid

    # ==================== TASK-LEVEL AUGMENTATION ====================

    def _apply_to_pair(self, pair: Dict, transform_fn) -> Dict:
        """Apply a transform to both input and output of a task pair."""
        return {
            "input": transform_fn(pair["input"]),
            "output": transform_fn(pair["output"]),
        }

    def _apply_color_permutation(self, task: Dict) -> Dict:
        """Apply random color permutation to entire task."""
        colors = list(range(10))
        self.rng.shuffle(colors)
        mapping = {i: colors[i] for i in range(10)}

        new_task = {"train": [], "test": []}
        for pair in task["train"]:
            new_task["train"].append({
                "input": self.permute_colors(pair["input"], mapping),
                "output": self.permute_colors(pair["output"], mapping),
            })
        for pair in task["test"]:
            new_task["test"].append({
                "input": self.permute_colors(pair["input"], mapping),
                "output": self.permute_colors(pair["output"], mapping),
            })
        return new_task

    def augment_task(self, task: Dict, num_variations: int = 500) -> List[Dict]:
        """Generate augmented versions of an ARC task."""
        augmented = []

        # Weighted strategy selection
        strategies = [
            ("color_only", 0.40),
            ("rotate_color", 0.15),
            ("flip_color", 0.15),
            ("scale_color", 0.10),
            ("noise_color", 0.10),
            ("translate_color", 0.05),
            ("multi_transform", 0.05),
        ]
        strategy_names = [s[0] for s in strategies]
        strategy_weights = [s[1] for s in strategies]

        for _ in range(num_variations):
            strategy = self.rng.choices(strategy_names, weights=strategy_weights, k=1)[0]

            try:
                if strategy == "color_only":
                    new_task = self._apply_color_permutation(task)

                elif strategy == "rotate_color":
                    rotations = self.rng.randint(1, 3)
                    transform = lambda g: self.rotate_90(g)  # noqa
                    for _ in range(rotations):
                        transform = lambda g, t=transform: self.rotate_90(t(g))  # noqa
                    new_task = copy.deepcopy(task)
                    new_task["train"] = [self._apply_to_pair(p, transform) for p in new_task["train"]]
                    new_task["test"] = [self._apply_to_pair(p, transform) for p in new_task["test"]]
                    new_task = self._apply_color_permutation(new_task)

                elif strategy == "flip_color":
                    flip = self.rng.choice([self.flip_horizontal, self.flip_vertical])
                    new_task = copy.deepcopy(task)
                    new_task["train"] = [self._apply_to_pair(p, flip) for p in new_task["train"]]
                    new_task["test"] = [self._apply_to_pair(p, flip) for p in new_task["test"]]
                    new_task = self._apply_color_permutation(new_task)

                elif strategy == "scale_color":
                    factor = self.rng.choice([2, 3])
                    new_task = copy.deepcopy(task)
                    new_task["train"] = [self._apply_to_pair(p, lambda g: self.scale_grid(g, factor)) for p in new_task["train"]]
                    new_task["test"] = [self._apply_to_pair(p, lambda g: self.scale_grid(g, factor)) for p in new_task["test"]]
                    new_task = self._apply_color_permutation(new_task)

                elif strategy == "noise_color":
                    new_task = copy.deepcopy(task)
                    new_task["train"] = [self._apply_to_pair(p, lambda g: self.add_noise(g, 2)) for p in new_task["train"]]
                    new_task["test"] = [self._apply_to_pair(p, lambda g: self.add_noise(g, 2)) for p in new_task["test"]]
                    new_task = self._apply_color_permutation(new_task)

                elif strategy == "translate_color":
                    dx = self.rng.randint(-3, 3)
                    dy = self.rng.randint(-3, 3)
                    new_task = copy.deepcopy(task)
                    new_task["train"] = [self._apply_to_pair(p, lambda g: self.translate_grid(g, dx, dy)) for p in new_task["train"]]
                    new_task["test"] = [self._apply_to_pair(p, lambda g: self.translate_grid(g, dx, dy)) for p in new_task["test"]]
                    new_task = self._apply_color_permutation(new_task)

                elif strategy == "multi_transform":
                    # Apply 2-3 random transforms
                    transforms_pool = [
                        self.rotate_90,
                        self.flip_horizontal,
                        self.flip_vertical,
                    ]
                    chosen = self.rng.sample(transforms_pool, k=self.rng.randint(2, 3))
                    new_task = copy.deepcopy(task)
                    for t_fn in chosen:
                        new_task["train"] = [self._apply_to_pair(p, t_fn) for p in new_task["train"]]
                        new_task["test"] = [self._apply_to_pair(p, t_fn) for p in new_task["test"]]
                    new_task = self._apply_color_permutation(new_task)

                else:
                    continue

                augmented.append(new_task)

            except Exception as e:
                # Some augmentations may fail (e.g., scaling very large grids)
                continue

        return augmented

    def grid_to_text(self, grid: List[List[int]]) -> str:
        """Convert grid to text representation."""
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    def format_as_code_reasoning(self, task: Dict, task_id: str = "") -> str:
        """Format an ARC task as a code-as-reasoning training example.

        The model learns to write Python code that transforms input to output.
        """
        prompt_parts = [
            "Given these input-output grid pairs, write a Python function that transforms the input grid to the output grid.\n"
        ]

        # Add training examples
        for i, pair in enumerate(task["train"]):
            prompt_parts.append(f"Example {i+1}:")
            prompt_parts.append("Input:")
            prompt_parts.append(self.grid_to_text(pair["input"]))
            prompt_parts.append("Output:")
            prompt_parts.append(self.grid_to_text(pair["output"]))
            prompt_parts.append("")

        # Add test input
        if task["test"]:
            prompt_parts.append("Test Input:")
            prompt_parts.append(self.grid_to_text(task["test"][0]["input"]))
            prompt_parts.append("")
            prompt_parts.append("Write a Python function `transform(grid)` that produces the output:")

        prompt = "\n".join(prompt_parts)

        # Completion (we don't have the actual Python code, but we format for training)
        if task["test"] and "output" in task["test"][0]:
            completion_parts = [
                "```python",
                "def transform(grid):",
                "    # Analyze the pattern from examples",
                f"    # Input shape: {len(task['test'][0]['input'])}x{len(task['test'][0]['input'][0])}",
                f"    # Output shape: {len(task['test'][0]['output'])}x{len(task['test'][0]['output'][0])}",
                "    result = [row[:] for row in grid]",
                "    # TODO: Apply transformation",
                "    return result",
                "```",
            ]
            completion = "\n".join(completion_parts)
        else:
            completion = ""

        return prompt + "\n" + completion

    def generate_dataset(
        self,
        arc1_path: Optional[str] = None,
        arc2_path: Optional[str] = None,
        variations_per_task: int = 500,
        output_path: str = "data/arc/augmented_arc.jsonl",
        format: str = "code_reasoning",
    ) -> None:
        """Generate the full augmented ARC dataset."""

        all_tasks = {}

        # Load ARC-AGI-1
        if arc1_path:
            all_tasks.update(self.load_tasks(arc1_path))

        # Load ARC-AGI-2
        if arc2_path:
            all_tasks.update(self.load_tasks(arc2_path))

        print(f"\nTotal tasks loaded: {len(all_tasks)}")
        print(f"Variations per task: {variations_per_task}")
        print(f"Expected output: ~{len(all_tasks) * variations_per_task} augmented tasks")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_generated = 0
        with open(output_path, "w") as f:
            for task_id, task in all_tasks.items():
                augmented = self.augment_task(task, num_variations=variations_per_task)

                for i, aug_task in enumerate(augmented):
                    if format == "code_reasoning":
                        text = self.format_as_code_reasoning(aug_task, task_id)
                        f.write(json.dumps({"text": text, "source": "arc", "task_id": task_id, "variation": i}) + "\n")
                    else:
                        f.write(json.dumps({
                            "task_id": task_id,
                            "variation": i,
                            "train": aug_task["train"],
                            "test": aug_task["test"],
                            "source": "arc",
                        }) + "\n")

                    total_generated += 1

                if total_generated % 10000 == 0:
                    print(f"  Generated {total_generated} augmented tasks...")

        print(f"\nDone! Generated {total_generated} augmented tasks")
        print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ZARX-1B ARC Augmentation Pipeline")
    parser.add_argument("--arc1_path", type=str, help="Path to ARC-AGI-1 training data")
    parser.add_argument("--arc2_path", type=str, help="Path to ARC-AGI-2 training data")
    parser.add_argument("--variations_per_task", type=int, default=500)
    parser.add_argument("--output_path", type=str, default="data/arc/augmented_arc.jsonl")
    parser.add_argument("--format", type=str, default="code_reasoning", choices=["code_reasoning", "raw"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augmenter = ARCAugmenter(seed=args.seed)
    augmenter.generate_dataset(
        arc1_path=args.arc1_path,
        arc2_path=args.arc2_path,
        variations_per_task=args.variations_per_task,
        output_path=args.output_path,
        format=args.format,
    )


if __name__ == "__main__":
    main()
