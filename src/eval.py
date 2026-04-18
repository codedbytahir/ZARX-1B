"""
ZARX-1B Evaluation Suite
Evaluates the model on code benchmarks and ARC tasks.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt."""
    model.eval()

    tokens = tokenizer.encode(prompt)
    if hasattr(tokens, 'ids'):
        input_ids = torch.tensor([tokens.ids], dtype=torch.long).to(device)
    else:
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_probs[sorted_indices_to_remove] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == tokenizer.token_to_id("<eos>"):
                break

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def evaluate_humaneval(
    model,
    tokenizer,
    num_samples: int = 10,
    temperature: float = 0.8,
    device: str = "cuda",
) -> Dict:
    """Evaluate on HumanEval benchmark (simplified version)."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        print("Could not load HumanEval dataset. Install: pip install datasets")
        return {"error": "dataset not available"}

    results = []
    correct = 0
    total = 0

    for problem in dataset:
        prompt = problem["prompt"]
        test = problem["test"]
        task_id = problem["task_id"]

        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=512,
            temperature=temperature,
            device=device,
        )

        # Extract code from generation
        completion = generated[len(prompt):]

        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "completion": completion,
        })

        total += 1
        if total % 10 == 0:
            print(f"Evaluated {total}/{len(dataset)} problems")

    return {
        "total": total,
        "pass_rate_estimate": f"{correct}/{total}",
        "results": results[:10],  # Sample
    }


def evaluate_arc(
    model,
    tokenizer,
    arc_path: str = "data/arc/arc-agi-1/training",
    max_tasks: int = 50,
    device: str = "cuda",
) -> Dict:
    """Evaluate on ARC-AGI tasks using program synthesis approach."""
    arc_dir = Path(arc_path)
    if not arc_dir.exists():
        return {"error": f"ARC data not found at {arc_path}"}

    tasks = []
    for f in sorted(arc_dir.glob("*.json"))[:max_tasks]:
        with open(f) as fp:
            tasks.append((f.stem, json.load(fp)))

    solved = 0
    total = len(tasks)

    for task_id, task in tasks:
        # Format task for model
        prompt = "Given these input-output grid pairs, write a Python function that transforms input to output.\n\n"
        for i, pair in enumerate(task["train"]):
            prompt += f"Example {i+1}:\nInput: {pair['input']}\nOutput: {pair['output']}\n\n"

        if task["test"]:
            prompt += f"Test Input: {task['test'][0]['input']}\n\nWrite the transform function:\n```python\ndef transform(grid):\n"

        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=512,
            temperature=0.2,
            device=device,
        )

        # Try to execute the generated code
        try:
            code = generated[len(prompt):]
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            exec_globals = {}
            exec(code, exec_globals)

            if "transform" in exec_globals:
                test_input = task["test"][0]["input"]
                result = exec_globals["transform"](test_input)
                expected = task["test"][0]["output"]

                if result == expected:
                    solved += 1
        except Exception:
            pass

    return {
        "total": total,
        "solved": solved,
        "accuracy": solved / total if total > 0 else 0,
    }


def quick_eval(model, tokenizer, device="cuda"):
    """Quick evaluation with sample generations."""
    test_prompts = [
        "def fibonacci(n):",
        "Write a Python function to reverse a linked list:",
        "Given these grid pairs, write a transform function:\nInput: [[1,0],[0,1]]\nOutput: [[0,1],[1,0]]\n\ndef transform(grid):",
        "class BinarySearchTree:",
        "def quicksort(arr):",
    ]

    print("=" * 60)
    print("ZARX-1B Quick Evaluation")
    print("=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:80]}...")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.2, device=device)
        print(f"Output: {generated[len(prompt):200]}")
        print("-" * 40)


if __name__ == "__main__":
    print("ZARX-1B Evaluation Suite")
    print("Run from training notebook after model is trained.")
