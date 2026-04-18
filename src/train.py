"""
ZARX-1B Training Script
Main training loop with auto-resume, gradient checkpointing, and W&B logging.

Designed to run on Google Colab T4 (free tier) and Kaggle T4/P100.
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import AdamW

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ZARXModel, ZARXConfig, build_model
from src.checkpoint_manager import CheckpointManager


class PretrainDataset(IterableDataset):
    """Streaming dataset for pretraining from JSONL files."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 2048, shuffle_buffer: int = 1000):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        import random
        buffer = []

        for jsonl_file in sorted(self.data_path.glob("*.jsonl")):
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        example = json.loads(line)
                        text = example.get("text", "")
                        if not text:
                            continue

                        # Tokenize
                        tokens = self.tokenizer.encode(text)
                        if isinstance(tokens, dict):
                            token_ids = tokens["input_ids"]
                        else:
                            token_ids = tokens.ids if hasattr(tokens, 'ids') else tokens

                        # Truncate to max_seq_len
                        token_ids = token_ids[:self.max_seq_len]

                        if len(token_ids) < 10:
                            continue

                        # Create input and labels (shift by 1)
                        input_ids = torch.tensor(token_ids, dtype=torch.long)
                        labels = input_ids.clone()

                        yield {
                            "input_ids": input_ids,
                            "labels": labels,
                            "attention_mask": torch.ones_like(input_ids),
                        }
                    except Exception:
                        continue


class PretrainDatasetFromHF(IterableDataset):
    """Streaming dataset from HuggingFace Hub."""

    def __init__(self, dataset_name: str, tokenizer, max_seq_len: int = 2048,
                 split: str = "train", text_field: str = "text"):
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field

    def __iter__(self):
        for example in self.dataset:
            text = example.get(self.text_field, "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, dict):
                token_ids = tokens["input_ids"]
            else:
                token_ids = tokens.ids if hasattr(tokens, 'ids') else tokens

            token_ids = token_ids[:self.max_seq_len]
            if len(token_ids) < 10:
                continue

            input_ids = torch.tensor(token_ids, dtype=torch.long)
            labels = input_ids.clone()

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids),
            }


def collate_fn(batch, max_seq_len=2048):
    """Pad and collate a batch of variable-length sequences."""
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        ids = item["input_ids"]
        lbl = item["labels"]
        mask = item["attention_mask"]
        pad_len = max_seq_len - len(ids)

        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        elif pad_len < 0:
            ids = ids[:max_seq_len]
            lbl = lbl[:max_seq_len]
            mask = mask[:max_seq_len]

        input_ids.append(ids)
        labels.append(lbl)
        attention_mask.append(mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    """Main training function."""

    # ==================== SETUP ====================

    print("=" * 60)
    print("  ZARX-1B Training")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load config
    config = ZARXConfig.from_json(args.model_config)
    print(f"\nModel Config:")
    print(f"  Vocab: {config.vocab_size}, Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}, Heads: {config.num_attention_heads}")
    print(f"  KV Heads (GQA): {config.num_key_value_heads}")
    print(f"  Context: {config.max_position_embeddings}")

    # Build model
    model = ZARXModel(config)
    model.tie_weights()
    model.enable_gradient_checkpointing()
    model.to(device)

    params = model.count_parameters()
    print(f"\nModel: {params['total_billion']:.3f}B parameters")
    print(f"Size in bf16: {params['total'] * 2 / 1e9:.2f} GB")

    # Load tokenizer
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        print(f"Tokenizer loaded from {args.tokenizer_path}")
    else:
        print("WARNING: No tokenizer found. Using character-level fallback.")
        tokenizer = None

    # ==================== DATA ====================

    if args.data_path:
        dataset = PretrainDataset(args.data_path, tokenizer, config.max_position_embeddings)
    elif args.hf_dataset:
        dataset = PretrainDatasetFromHF(
            args.hf_dataset, tokenizer, config.max_position_embeddings,
            split=args.data_split, text_field=args.text_field,
        )
    else:
        raise ValueError("Must provide --data_path or --hf_dataset")

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        collate_fn=lambda b: collate_fn(b, config.max_position_embeddings),
        num_workers=0,  # Colab/Kaggle work better with 0 workers
        pin_memory=True,
    )

    # ==================== OPTIMIZER ====================

    if BNB_AVAILABLE and args.use_8bit_adam:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
        print("Using 8-bit AdamW (bitsandbytes)")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
        print("Using standard AdamW")

    # ==================== CHECKPOINT MANAGER ====================

    ckpt_mgr = CheckpointManager(
        local_dir=args.checkpoint_dir,
        drive_dir=args.drive_dir,
        github_token=args.github_token,
        github_repo=args.github_repo,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        save_every_local=args.save_every_local,
        save_every_drive=args.save_every_drive,
        save_every_github=args.save_every_github,
        save_every_hf=args.save_every_hf,
    )

    # Resume from checkpoint
    start_step = 0
    total_tokens = 0
    checkpoint = ckpt_mgr.load_latest()
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        total_tokens = checkpoint.get("total_tokens", 0)
        print(f"\nResumed from step {start_step}, {total_tokens/1e9:.2f}B tokens seen")
        del checkpoint
        torch.cuda.empty_cache()

    # ==================== SCHEDULER ====================

    total_steps = args.total_tokens // (args.micro_batch_size * args.gradient_accumulation_steps * config.max_position_embeddings)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,
    )

    # ==================== WANDB ====================

    if WANDB_AVAILABLE and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"zarx-1b-pretrain",
            config={
                "model": config.__dict__,
                "training": vars(args),
            },
            resume="allow",
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # ==================== TRAINING LOOP ====================

    model.train()
    gradient_accumulation_steps = args.gradient_accumulation_steps
    log_every = args.log_every_steps

    print(f"\nTraining Configuration:")
    print(f"  Total steps:     {total_steps}")
    print(f"  Start step:      {start_step}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Batch size:      {args.micro_batch_size} x {gradient_accumulation_steps} accum = {args.micro_batch_size * gradient_accumulation_steps} effective")
    print(f"  Tokens/step:     {args.micro_batch_size * gradient_accumulation_steps * config.max_position_embeddings}")
    print(f"  Target tokens:   {args.total_tokens/1e9:.1f}B")
    print(f"  Warmup steps:    {args.warmup_steps}")
    print(f"\nStarting training...\n")

    data_iter = iter(dataloader)
    running_loss = 0.0
    step_start_time = time.time()

    for step in range(start_step, total_steps):
        # Accumulate gradients
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs["loss"] / gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Track metrics
        current_loss = accumulated_loss
        running_loss += current_loss
        tokens_this_step = input_ids.numel() * gradient_accumulation_steps
        total_tokens += tokens_this_step

        # Logging
        if step % log_every == 0 and step > 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - step_start_time
            tokens_per_sec = (tokens_this_step * log_every) / elapsed
            lr = scheduler.get_last_lr()[0]
            progress = step / total_steps * 100

            print(
                f"Step {step:>7d}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tokens: {total_tokens/1e9:.2f}B | "
                f"Speed: {tokens_per_sec:.0f} tok/s | "
                f"Progress: {progress:.1f}% | "
                f"ETA: {elapsed/log_every * (total_steps-step) / 3600:.1f}h"
            )

            if WANDB_AVAILABLE and args.wandb_project and wandb.run:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/tokens": total_tokens,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                    "train/progress": progress,
                })

            running_loss = 0.0
            step_start_time = time.time()

        # Checkpoint
        ckpt_mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            total_tokens=total_tokens,
            train_loss=current_loss,
        )

    # ==================== FINAL SAVE ====================

    print("\nTraining complete! Saving final model...")
    ckpt_mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=total_steps,
        total_tokens=total_tokens,
        train_loss=current_loss,
    )

    if args.output_dir:
        model.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

    if WANDB_AVAILABLE and args.wandb_project and wandb.run:
        wandb.finish()

    print("\nZARX-1B training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="ZARX-1B Training")

    # Model
    parser.add_argument("--model_config", type=str, default="configs/model_config.json")
    parser.add_argument("--tokenizer_path", type=str, default="configs/tokenizer.json")

    # Data
    parser.add_argument("--data_path", type=str, default=None, help="Local JSONL data path")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")

    # Training
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--total_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="/content/checkpoints")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/ZARX-1B")
    parser.add_argument("--hf_repo_id", type=str, default="Chvigo/zarx-checks")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--github_token", type=str, default=None, help="GitHub personal token for checkpoint pushes")
    parser.add_argument("--github_repo", type=str, default="codedbytahir/zarx-checkpoints", help="GitHub checkpoint repo")
    parser.add_argument("--save_every_local", type=int, default=100)
    parser.add_argument("--save_every_drive", type=int, default=500)
    parser.add_argument("--save_every_github", type=int, default=1000)
    parser.add_argument("--save_every_hf", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="/content/zarx-1b-final")

    # Logging
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="zarx-1b")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
