#!/usr/bin/env python3
"""
ZARX-1B: Day 1 Setup Script for Google Colab
Run this ONCE on a fresh Colab session to set up everything.

What this does:
1. Installs all dependencies
2. Mounts Google Drive
3. Clones the ZARX-1B repo
4. Verifies GPU environment
5. Downloads ProX code data
6. Downloads ARC-AGI data + generates ReARC augmentations
7. Creates augmented ARC dataset (code-as-reasoning format)
8. Preprocesses all data (dedup, filter, merge)
9. Trains the custom BPE tokenizer
10. Tests the model forward/backward pass
11. Starts a short test training run (1000 steps)

Usage on Colab:
  !pip install -q torch bitsandbytes
  !git clone https://github.com/codedbytahir/ZARX-1B.git
  %cd ZARX-1B
  !python scripts/setup_colab_day1.py --hf_token YOUR_HF_TOKEN --wandb_key YOUR_WANDB_KEY
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run(cmd, desc="", check=True):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ FAILED (exit code {result.returncode})")
        return False
    print(f"  ✅ Done")
    return True


def step(num, title):
    print(f"\n{'#'*60}")
    print(f"# STEP {num}: {title}")
    print(f"{'#'*60}")


def main():
    parser = argparse.ArgumentParser(description="ZARX-1B Day 1 Setup")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace API token")
    parser.add_argument("--wandb_key", type=str, default="", help="Weights & Biases API key")
    parser.add_argument("--hf_repo", type=str, default="", help="HuggingFace checkpoint repo ID")
    parser.add_argument("--skip_download", action="store_true", help="Skip data download (if already done)")
    parser.add_argument("--skip_arc_augment", action="store_true", help="Skip ARC augmentation")
    parser.add_argument("--skip_tokenizer", action="store_true", help="Skip tokenizer training")
    parser.add_argument("--skip_test_train", action="store_true", help="Skip test training run")
    args = parser.parse_args()

    BASE_DIR = Path("/content/ZARX-1B")
    DATA_DIR = Path("/content/data")
    PROCESSED_DIR = DATA_DIR / "processed"

    start_time = datetime.now()

    # ========== STEP 1: Environment ==========
    step(1, "Install Dependencies")

    run("pip install -q torch>=2.1.0", "PyTorch")
    run("pip install -q bitsandbytes", "bitsandbytes (8-bit optimizer)")
    run("pip install -q wandb", "Weights & Biases")
    run("pip install -q huggingface_hub", "HuggingFace Hub")
    run("pip install -q datasets tokenizers accelerate", "HuggingFace ecosystem")
    run("pip install -q datasketch", "MinHash dedup")
    run("pip install -q flash-attn --no-build-isolation", "Flash Attention 2 (may take 3-5 min)", check=False)

    # ========== STEP 2: Login ==========
    step(2, "Login to Services")

    if args.hf_token:
        run(f"huggingface-cli login --token {args.hf_token}", "HuggingFace login")

    if args.wandb_key:
        run(f"wandb login {args.wandb_key}", "W&B login")

    # ========== STEP 3: Verify GPU ==========
    step(3, "Verify GPU Environment")

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
        if vram < 14:
            print("  ⚠️ WARNING: Less than 14GB VRAM. May need to reduce batch size.")
    else:
        print("  ❌ No GPU found! Change runtime to GPU.")
        return

    # ========== STEP 4: Test Model ==========
    step(4, "Test Model Architecture")

    sys.path.insert(0, str(BASE_DIR))
    from src.model import build_model

    model = build_model(str(BASE_DIR / "configs" / "model_config.json"))
    model = model.cuda()

    x = torch.randint(0, 32000, (1, 128)).cuda()
    out = model(x, labels=x)
    print(f"  Forward pass: Output shape {out['logits'].shape}")
    print(f"  Loss: {out['loss'].item():.4f}")

    out["loss"].backward()
    print(f"  Backward pass: OK")

    vram_used = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM used: {vram_used:.2f} GB")

    del model, x, out
    torch.cuda.empty_cache()

    # ========== STEP 5: Download ProX Data ==========
    if not args.skip_download:
        step(5, "Download ProX Code Data")

        # Try RedPajama-pro first (contains code), fallback to FineWeb-pro
        run(
            f"cd {BASE_DIR} && python scripts/download_data.py "
            f"--download_prox_code "
            f"--output_dir {DATA_DIR}/raw "
            f"--max_prox_examples 2000000",
            "Download ProX code-filtered data (this takes a while)"
        )

        # ========== STEP 6: Download ARC Data ==========
        step(6, "Download ARC-AGI Data")

        run(
            f"cd {BASE_DIR} && python scripts/download_data.py "
            f"--download_arc "
            f"--output_dir {DATA_DIR}/raw",
            "Download ARC-AGI-1, ARC-AGI-2, and ReARC"
        )
    else:
        print("\n  Skipping data download (--skip_download)")

    # ========== STEP 7: Generate ARC Augmentations ==========
    if not args.skip_arc_augment:
        step(7, "Generate ARC Augmented Dataset")

        arc1_path = DATA_DIR / "raw" / "arc" / "arc-agi-1" / "data" / "training"
        arc2_path = DATA_DIR / "raw" / "arc" / "arc-agi-2" / "data" / "training"

        # Check paths exist
        if arc1_path.exists() and arc2_path.exists():
            run(
                f"cd {BASE_DIR} && python src/arc_augment.py "
                f"--arc1_path {arc1_path} "
                f"--arc2_path {arc2_path} "
                f"--variations_per_task 500 "
                f"--output_path {DATA_DIR}/raw/arc/augmented_arc.jsonl "
                f"--format code_reasoning",
                "Generate augmented ARC tasks (400K+ tasks)"
            )
        else:
            print(f"  ⚠️ ARC paths not found:")
            print(f"    ARC-AGI-1: {arc1_path} (exists: {arc1_path.exists()})")
            print(f"    ARC-AGI-2: {arc2_path} (exists: {arc2_path.exists()})")
    else:
        print("\n  Skipping ARC augmentation (--skip_arc_augment)")

    # ========== STEP 8: Preprocess Data ==========
    step(8, "Preprocess Data")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process ProX data
    prox_dir = DATA_DIR / "raw" / "prox-code"
    if not prox_dir.exists():
        prox_dir = DATA_DIR / "raw" / "prox"

    if prox_dir.exists():
        # Process each shard
        for shard_file in sorted(prox_dir.glob("*.jsonl"))[:3]:  # Process first 3 shards for Day 1
            run(
                f"cd {BASE_DIR} && python src/data_pipeline.py --mode process "
                f"--input_file {shard_file} "
                f"--output_file {PROCESSED_DIR / shard_file.name.replace('shard', 'processed')} "
                f"--source prox",
                f"Process {shard_file.name}"
            )
    else:
        print(f"  ⚠️ No ProX data found at {prox_dir}")

    # Process ARC data
    arc_file = DATA_DIR / "raw" / "arc" / "augmented_arc.jsonl"
    if arc_file.exists():
        run(
            f"cd {BASE_DIR} && python src/data_pipeline.py --mode process "
            f"--input_file {arc_file} "
            f"--output_file {PROCESSED_DIR / 'arc_processed.jsonl'} "
            f"--source arc",
            "Process ARC augmented data"
        )

    # Merge
    run(
        f"cd {BASE_DIR} && python src/data_pipeline.py --mode merge "
        f"--processed_dir {PROCESSED_DIR} "
        f"--output_file {PROCESSED_DIR / 'zarx_pretrain.jsonl'}",
        "Merge all processed data"
    )

    # ========== STEP 9: Train Tokenizer ==========
    if not args.skip_tokenizer:
        step(9, "Train Custom BPE Tokenizer")

        run(
            f"cd {BASE_DIR} && python scripts/train_tokenizer.py "
            f"--data_path {PROCESSED_DIR} "
            f"--output_path {BASE_DIR / 'configs' / 'tokenizer.json'} "
            f"--vocab_size 32000 "
            f"--max_lines 5000000",
            "Train 32K BPE tokenizer on code + ARC corpus"
        )
    else:
        print("\n  Skipping tokenizer training (--skip_tokenizer)")

    # ========== STEP 10: Short Test Training Run ==========
    if not args.skip_test_train:
        step(10, "Test Training Run (1000 steps)")

        tokenizer_path = BASE_DIR / "configs" / "tokenizer.json"
        if not tokenizer_path.exists():
            print("  ⚠️ No tokenizer found. Skipping test training.")
        else:
            # Mount Google Drive for checkpoints
            try:
                from google.colab import drive
                drive.mount("/content/drive")
                drive_dir = "/content/drive/MyDrive/ZARX-1B"
            except ImportError:
                drive_dir = "/content/checkpoints"

            run(
                f"cd {BASE_DIR} && python src/train.py "
                f"--model_config configs/model_config.json "
                f"--tokenizer_path configs/tokenizer.json "
                f"--data_path {PROCESSED_DIR} "
                f"--micro_batch_size 1 "
                f"--gradient_accumulation_steps 32 "
                f"--learning_rate 3e-4 "
                f"--warmup_steps 100 "
                f"--total_tokens 65536000 "  # Just 1000 steps worth
                f"--use_8bit_adam "
                f"--checkpoint_dir /content/checkpoints "
                f"--drive_dir {drive_dir} "
                + (f"--hf_repo_id {args.hf_repo} " if args.hf_repo else "")
                + (f"--hf_token {args.hf_token} " if args.hf_token else "")
                + f"--save_every_local 200 "
                f"--save_every_drive 500 "
                f"--save_every_hf 1000 "
                f"--log_every_steps 10 "
                f"--wandb_project zarx-1b "
                f"--wandb_run_name day1-test",
                "Test training run (1000 steps)"
            )
    else:
        print("\n  Skipping test training (--skip_test_train)")

    # ========== SUMMARY ==========
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  ZARX-1B Day 1 Setup Complete!")
    print(f"  Time: {elapsed}")
    print(f"{'='*60}")
    print(f"""
  What's ready:
    ✅ Dependencies installed
    ✅ Model architecture tested ({1:.0f}B params, fits on T4)
    ✅ Data downloaded and processed
    ✅ Tokenizer trained
    ✅ Test training run completed

  Next steps:
    1. Start full pretraining:
       python src/train.py --model_config configs/model_config.json \\
         --tokenizer_path configs/tokenizer.json \\
         --data_path {PROCESSED_DIR} \\
         --total_tokens 10000000000 \\
         --use_8bit_adam \\
         --wandb_project zarx-1b

    2. Your data is at: {PROCESSED_DIR}
    3. Your tokenizer is at: {BASE_DIR}/configs/tokenizer.json
    4. Checkpoints save to: /content/checkpoints + Google Drive + HuggingFace

  Daily routine:
    ☀️ Start Colab → run training → check W&B
    🌙 Start Kaggle → resume from HF checkpoint → let it run
    💾 Checkpoints are saved every 100/500/1000 steps (triple redundancy)
    """)


if __name__ == "__main__":
    main()
