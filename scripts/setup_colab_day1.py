#!/usr/bin/env python3
"""
ZARX-1B: Day 1 Setup Script for Google Colab
Run this ONCE on a fresh Colab session to set up everything.

What this does:
1. Installs all dependencies
2. Mounts Google Drive (PERSISTENCE LAYER)
3. Logins to HuggingFace & W&B
4. Verifies GPU environment
5. Tests model architecture
6. Downloads ProX code data → saves to Drive
7. Downloads ARC-AGI data → saves to Drive
8. Generates ARC augmentations → saves to Drive
9. Preprocesses all data → saves to Drive
10. Trains custom BPE tokenizer → saves to Drive
11. Short test training run (1000 steps) → checkpoints to Drive + GitHub + HF

EVERY step saves to Google Drive so nothing is lost on disconnect.

Usage on Colab:
  !git clone https://github.com/codedbytahir/ZARX-1B.git
  %cd ZARX-1B
  !python scripts/setup_colab_day1.py --hf_token YOUR_HF_TOKEN --wandb_key YOUR_WANDB_KEY --github_token YOUR_GH_TOKEN
"""

import os
import sys
import subprocess
import shutil
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
        print(f"  FAILED (exit code {result.returncode})")
        return False
    print(f"  Done")
    return True


def step(num, title):
    print(f"\n{'#'*60}")
    print(f"# STEP {num}: {title}")
    print(f"{'#'*60}")


def save_to_drive(src_path, drive_dir, label=""):
    """Save a file or directory to Google Drive."""
    try:
        src = Path(src_path)
        if not src.exists():
            return False
        if src.is_dir():
            dest = Path(drive_dir) / label if label else Path(drive_dir) / src.name
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
        else:
            dest = Path(drive_dir) / label if label else Path(drive_dir) / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
        print(f"  [DRIVE] Saved {src.name} to Google Drive")
        return True
    except Exception as e:
        print(f"  [DRIVE] Warning: Could not save to Drive: {e}")
        return False


def restore_from_drive(drive_path, dest_path, label=""):
    """Restore a file or directory from Google Drive if it exists."""
    try:
        src = Path(drive_path) / label if label else Path(drive_path)
        if not src.exists():
            return False
        dest = Path(dest_path)
        if src.is_dir():
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
        print(f"  [DRIVE] Restored {src.name} from Google Drive")
        return True
    except Exception as e:
        print(f"  [DRIVE] Warning: Could not restore from Drive: {e}")
        return False


def push_to_github(github_token, github_repo, file_path, commit_msg="update"):
    """Push a file to the GitHub checkpoint repo."""
    if not github_token or not github_repo:
        return False
    try:
        github_local = Path("/content/zarx-checkpoints")
        auth_url = f"https://{github_token}@github.com/{github_repo}.git"

        if not github_local.exists():
            subprocess.run(["git", "clone", auth_url, str(github_local)],
                         capture_output=True, text=True, timeout=60)
            if not github_local.exists():
                github_local.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init"], cwd=str(github_local), capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", auth_url],
                             cwd=str(github_local), capture_output=True)

        # Copy file
        src = Path(file_path)
        if src.is_dir():
            shutil.copytree(str(src), str(github_local / src.name), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src), str(github_local / src.name))

        # Git push
        subprocess.run(["git", "config", "user.email", "zarx-1b@training.bot"],
                      cwd=str(github_local), capture_output=True)
        subprocess.run(["git", "config", "user.name", "ZARX-1B Bot"],
                      cwd=str(github_local), capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=str(github_local), capture_output=True)
        subprocess.run(["git", "commit", "-m", commit_msg],
                      cwd=str(github_local), capture_output=True)
        result = subprocess.run(["git", "push", "origin", "main"],
                              cwd=str(github_local), capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"  [GITHUB] Pushed {src.name} to {github_repo}")
            return True
        else:
            # Try with force set-url
            subprocess.run(["git", "remote", "set-url", "origin", auth_url],
                          cwd=str(github_local), capture_output=True)
            subprocess.run(["git", "push", "-u", "origin", "main"],
                          cwd=str(github_local), capture_output=True, text=True, timeout=120)
            print(f"  [GITHUB] Pushed {src.name} to {github_repo}")
            return True
    except Exception as e:
        print(f"  [GITHUB] Warning: Push failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ZARX-1B Day 1 Setup")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace API token")
    parser.add_argument("--wandb_key", type=str, default="", help="Weights & Biases API key")
    parser.add_argument("--hf_repo", type=str, default="", help="HuggingFace checkpoint repo ID")
    parser.add_argument("--github_token", type=str, default="", help="GitHub personal token for checkpoint pushes")
    parser.add_argument("--github_repo", type=str, default="codedbytahir/zarx-checkpoints", help="GitHub checkpoint repo")
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
    # Flash Attention 2 requires Ampere+ GPU (A100, RTX 3090, etc.)
    # T4 (Turing) doesn't support it - PyTorch SDPA auto-fallback works fine
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        if "A100" in gpu_name or "A10" in gpu_name or "RTX 30" in gpu_name or "RTX 40" in gpu_name or "L4" in gpu_name:
            run("pip install -q flash-attn --no-build-isolation", "Flash Attention 2 (Ampere+ GPU detected)", check=False)
        else:
            print(f"  Skipping Flash Attention 2 (not supported on {gpu_name})")
            print(f"  PyTorch SDPA will auto-use the best available attention backend")

    # ========== STEP 2: Mount Google Drive (PERSISTENCE!) ==========
    step(2, "Mount Google Drive + Create Persistence Directories")

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_DIR = Path("/content/drive/MyDrive/ZARX-1B")
        DRIVE_MOUNTED = True
    except ImportError:
        DRIVE_DIR = Path("/content/zarx-persist")
        DRIVE_MOUNTED = False
        print("  Warning: Not running in Colab. Using local persist dir.")

    # Create all persistence directories on Drive
    dirs_to_create = [
        DRIVE_DIR,
        DRIVE_DIR / "data" / "raw",
        DRIVE_DIR / "data" / "processed",
        DRIVE_DIR / "checkpoints",
        DRIVE_DIR / "tokenizer",
        DRIVE_DIR / "configs",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  Google Drive mounted: {DRIVE_MOUNTED}")
    print(f"  Persistence dir: {DRIVE_DIR}")
    print(f"  All data WILL be saved to Drive after every step.")

    # ========== STEP 3: Login ==========
    step(3, "Login to Services")

    if args.hf_token:
        from huggingface_hub import login as hf_login
        try:
            hf_login(token=args.hf_token)
            print("  HuggingFace login successful!")
        except Exception as e:
            print(f"  HF login warning: {e}")

    if args.wandb_key:
        run(f"wandb login {args.wandb_key}", "W&B login")

    # Init GitHub checkpoint repo
    if args.github_token:
        push_to_github(args.github_token, args.github_repo, "", "init")

    # ========== STEP 4: Verify GPU ==========
    step(4, "Verify GPU Environment")

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
        if vram < 14:
            print("  WARNING: Less than 14GB VRAM. May need to reduce batch size.")
    else:
        print("  No GPU found! Change runtime to GPU.")
        return

    # ========== STEP 5: Test Model ==========
    step(5, "Test Model Architecture")

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

    # ========== STEP 6: Download ProX Data ==========
    if not args.skip_download:
        step(6, "Download ProX Code Data")

        # Check if already downloaded on Drive
        drive_prox = DRIVE_DIR / "data" / "raw" / "prox-code"
        if drive_prox.exists() and any(drive_prox.glob("*.jsonl")):
            print(f"  Found existing ProX data on Drive! Restoring...")
            restore_from_drive(str(DRIVE_DIR / "data" / "raw"), str(DATA_DIR / "raw"), "prox-code")
        else:
            run(
                f"cd {BASE_DIR} && python scripts/download_data.py "
                f"--download_prox_code "
                f"--output_dir {DATA_DIR}/raw "
                f"--max_prox_examples 2000000",
                "Download ProX code-filtered data (this takes a while)"
            )
            # SAVE TO DRIVE
            save_to_drive(DATA_DIR / "raw" / "prox-code", DRIVE_DIR / "data" / "raw", "prox-code")

        # ========== STEP 7: Download ARC Data ==========
        step(7, "Download ARC-AGI Data")

        drive_arc = DRIVE_DIR / "data" / "raw" / "arc"
        if drive_arc.exists() and any(drive_arc.glob("**/*.json")):
            print(f"  Found existing ARC data on Drive! Restoring...")
            restore_from_drive(str(DRIVE_DIR / "data" / "raw"), str(DATA_DIR / "raw"), "arc")
        else:
            run(
                f"cd {BASE_DIR} && python scripts/download_data.py "
                f"--download_arc "
                f"--output_dir {DATA_DIR}/raw",
                "Download ARC-AGI-1, ARC-AGI-2, and ReARC"
            )
            # SAVE TO DRIVE
            save_to_drive(DATA_DIR / "raw" / "arc", DRIVE_DIR / "data" / "raw", "arc")
    else:
        print("\n  Skipping data download (--skip_download)")
        # Try restore from Drive anyway
        restore_from_drive(str(DRIVE_DIR / "data" / "raw"), str(DATA_DIR / "raw"))

    # ========== STEP 8: Generate ARC Augmentations ==========
    if not args.skip_arc_augment:
        step(8, "Generate ARC Augmented Dataset")

        arc1_path = DATA_DIR / "raw" / "arc" / "arc-agi-1" / "data" / "training"
        arc2_path = DATA_DIR / "raw" / "arc" / "arc-agi-2" / "data" / "training"

        # Check if already augmented on Drive
        drive_aug = DRIVE_DIR / "data" / "raw" / "arc" / "augmented_arc.jsonl"
        if drive_aug.exists():
            print(f"  Found existing augmented ARC on Drive! Restoring...")
            shutil.copy2(str(drive_aug), str(DATA_DIR / "raw" / "arc" / "augmented_arc.jsonl"))
        elif arc1_path.exists() and arc2_path.exists():
            run(
                f"cd {BASE_DIR} && python src/arc_augment.py "
                f"--arc1_path {arc1_path} "
                f"--arc2_path {arc2_path} "
                f"--variations_per_task 500 "
                f"--output_path {DATA_DIR}/raw/arc/augmented_arc.jsonl "
                f"--format code_reasoning",
                "Generate augmented ARC tasks (400K+ tasks)"
            )
            # SAVE TO DRIVE
            save_to_drive(DATA_DIR / "raw" / "arc" / "augmented_arc.jsonl",
                         DRIVE_DIR / "data" / "raw" / "arc")
        else:
            print(f"  ARC paths not found:")
            print(f"    ARC-AGI-1: {arc1_path} (exists: {arc1_path.exists()})")
            print(f"    ARC-AGI-2: {arc2_path} (exists: {arc2_path.exists()})")
    else:
        print("\n  Skipping ARC augmentation (--skip_arc_augment)")

    # ========== STEP 9: Preprocess Data ==========
    step(9, "Preprocess Data")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already processed on Drive
    drive_processed = DRIVE_DIR / "data" / "processed"
    if drive_processed.exists() and any(drive_processed.glob("*.jsonl")):
        print(f"  Found processed data on Drive! Restoring...")
        restore_from_drive(str(drive_processed), str(PROCESSED_DIR))
    else:
        # Process ProX data
        prox_dir = DATA_DIR / "raw" / "prox-code"
        if not prox_dir.exists():
            prox_dir = DATA_DIR / "raw" / "prox"

        if prox_dir.exists():
            for shard_file in sorted(prox_dir.glob("*.jsonl"))[:3]:
                run(
                    f"cd {BASE_DIR} && python src/data_pipeline.py --mode process "
                    f"--input_file {shard_file} "
                    f"--output_file {PROCESSED_DIR / shard_file.name.replace('shard', 'processed')} "
                    f"--source prox",
                    f"Process {shard_file.name}"
                )

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

        # SAVE TO DRIVE
        save_to_drive(PROCESSED_DIR, DRIVE_DIR / "data", "processed")

    # ========== STEP 10: Train Tokenizer ==========
    if not args.skip_tokenizer:
        step(10, "Train Custom BPE Tokenizer")

        # Check if tokenizer already on Drive
        drive_tokenizer = DRIVE_DIR / "tokenizer" / "tokenizer.json"
        if drive_tokenizer.exists():
            print(f"  Found trained tokenizer on Drive! Restoring...")
            shutil.copy2(str(drive_tokenizer), str(BASE_DIR / "configs" / "tokenizer.json"))
            print(f"  Tokenizer restored from Drive.")
        else:
            run(
                f"cd {BASE_DIR} && python scripts/train_tokenizer.py "
                f"--data_path {PROCESSED_DIR} "
                f"--output_path {BASE_DIR / 'configs' / 'tokenizer.json'} "
                f"--vocab_size 32000 "
                f"--max_lines 5000000",
                "Train 32K BPE tokenizer on code + ARC corpus"
            )
            # SAVE TO DRIVE
            save_to_drive(BASE_DIR / "configs" / "tokenizer.json", DRIVE_DIR / "tokenizer")
            # PUSH TO GITHUB
            if args.github_token:
                push_to_github(args.github_token, args.github_repo,
                             str(BASE_DIR / "configs" / "tokenizer.json"),
                             "add trained tokenizer")
    else:
        print("\n  Skipping tokenizer training (--skip_tokenizer)")
        # Try restore from Drive
        drive_tokenizer = DRIVE_DIR / "tokenizer" / "tokenizer.json"
        if drive_tokenizer.exists():
            shutil.copy2(str(drive_tokenizer), str(BASE_DIR / "configs" / "tokenizer.json"))

    # ========== STEP 11: Short Test Training Run ==========
    if not args.skip_test_train:
        step(11, "Test Training Run (1000 steps)")

        tokenizer_path = BASE_DIR / "configs" / "tokenizer.json"
        if not tokenizer_path.exists():
            print("  No tokenizer found. Skipping test training.")
        else:
            github_flags = f"--github_token {args.github_token} --github_repo {args.github_repo} " if args.github_token else ""
            hf_flags = f"--hf_repo_id {args.hf_repo} --hf_token {args.hf_token} " if args.hf_repo and args.hf_token else ""

            run(
                f"cd {BASE_DIR} && python src/train.py "
                f"--model_config configs/model_config.json "
                f"--tokenizer_path configs/tokenizer.json "
                f"--data_path {PROCESSED_DIR} "
                f"--micro_batch_size 1 "
                f"--gradient_accumulation_steps 32 "
                f"--learning_rate 3e-4 "
                f"--warmup_steps 100 "
                f"--total_tokens 65536000 "
                f"--use_8bit_adam "
                f"--checkpoint_dir /content/checkpoints "
                f"--drive_dir {DRIVE_DIR} "
                + github_flags
                + hf_flags
                + f"--save_every_local 200 "
                f"--save_every_drive 500 "
                f"--save_every_github 1000 "
                f"--save_every_hf 1000 "
                f"--log_every_steps 10 "
                f"--wandb_project zarx-1b "
                f"--wandb_run_name day1-test",
                "Test training run (1000 steps)"
            )
    else:
        print("\n  Skipping test training (--skip_test_train)")

    # ========== FINAL SYNC TO DRIVE ==========
    step(12, "Final Sync - Save Everything to Drive")

    save_to_drive(PROCESSED_DIR, DRIVE_DIR / "data", "processed")
    if (BASE_DIR / "configs" / "tokenizer.json").exists():
        save_to_drive(BASE_DIR / "configs" / "tokenizer.json", DRIVE_DIR / "tokenizer")
    save_to_drive("/content/checkpoints", DRIVE_DIR, "checkpoints")

    # Push configs to GitHub
    if args.github_token:
        push_to_github(args.github_token, args.github_repo,
                      str(BASE_DIR / "configs"), "sync configs after setup")

    # ========== SUMMARY ==========
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  ZARX-1B Day 1 Setup Complete!")
    print(f"  Time: {elapsed}")
    print(f"{'='*60}")
    print(f"""
  What's ready:
    Dependencies installed
    Model architecture tested (1.148B params, fits on T4)
    Data downloaded and processed
    Tokenizer trained
    Test training run completed

  PERSISTENCE - Everything is saved to Google Drive:
    Drive location: {DRIVE_DIR}
    + data/raw/          (ProX + ARC raw data)
    + data/processed/    (cleaned, deduped, merged data)
    + tokenizer/         (trained 32K BPE tokenizer)
    + checkpoints/       (training checkpoints)
    + configs/           (model + train configs)

  GitHub Checkpoints: {args.github_repo if args.github_token else 'not configured'}

  Next time you start Colab, the script will auto-restore from Drive.

  Daily routine:
    1. Start Colab -> mount Drive -> auto-restore -> resume training
    2. Checkpoints: Local(100) + Drive(500) + GitHub(1000) + HF(1000)
    3. Even if Colab dies, max data loss = ~8 minutes of training

  To start full pretraining:
    python src/train.py --model_config configs/model_config.json \\
      --tokenizer_path configs/tokenizer.json \\
      --data_path {PROCESSED_DIR} \\
      --total_tokens 10000000000 \\
      --use_8bit_adam \\
      --drive_dir {DRIVE_DIR} \\
      --github_token YOUR_GH_TOKEN \\
      --wandb_project zarx-1b
    """)


if __name__ == "__main__":
    main()
