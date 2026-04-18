#!/usr/bin/env python3
"""
ZARX-1B: Day 1 Setup Script for Google Colab

*** IMPORTANT: You MUST mount Google Drive in a notebook cell BEFORE running this script! ***

Run these cells first in your Colab notebook:

  Cell 1:
    from google.colab import drive
    drive.mount('/content/drive')

  Cell 2:
    !git clone https://github.com/codedbytahir/ZARX-1B.git
    %cd ZARX-1B
    !python scripts/setup_colab_day1.py --hf_token YOUR_HF_TOKEN --wandb_key YOUR_WANDB_KEY --github_token YOUR_GH_TOKEN

Persistence Strategy (Drive is OPTIONAL - never blocks the process):
  1. Try Google Drive first (if available + has space)
  2. Fall back to GitHub repo (always works if token provided)
  3. Fall back to HuggingFace (if token provided)
  4. Local disk always works within the session
  THE PROCESS NEVER STOPS due to storage failures.
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime


# Track Drive availability globally
DRIVE_WORKING = True


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
    """Save to Drive. NEVER raises - always returns True/False. Non-blocking."""
    global DRIVE_WORKING
    if not DRIVE_WORKING:
        print(f"  [DRIVE] Skipped (Drive previously failed)")
        return False
    try:
        src = Path(src_path)
        if not src.exists():
            return False

        # Check Drive space before attempting
        drive_path = Path(drive_dir)
        if drive_path.exists():
            stat = shutil.disk_usage(str(drive_path))
            free_gb = stat.free / 1e9
            src_size = sum(f.stat().st_size for f in src.rglob('*') if f.is_file()) if src.is_dir() else src.stat().st_size
            src_gb = src_size / 1e9
            if free_gb < src_gb + 0.5:  # Need at least 500MB buffer
                print(f"  [DRIVE] Not enough space ({free_gb:.1f}GB free, need {src_gb:.1f}GB). Skipping Drive.")
                DRIVE_WORKING = False
                return False

        if src.is_dir():
            dest = Path(drive_dir) / label if label else Path(drive_dir) / src.name
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
        else:
            dest = Path(drive_dir) / label if label else Path(drive_dir) / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
        print(f"  [DRIVE] Saved {src.name} -> Google Drive")
        return True
    except OSError as e:
        if "No space left" in str(e) or "Quota exceeded" in str(e):
            print(f"  [DRIVE] DRIVE IS FULL! Disabling Drive saves for rest of session.")
            DRIVE_WORKING = False
        else:
            print(f"  [DRIVE] Warning: Could not save to Drive: {e}")
        return False
    except Exception as e:
        print(f"  [DRIVE] Warning: Could not save to Drive: {e}")
        return False


def persist(src_path, drive_dir, github_token="", github_repo="", hf_token="", hf_repo="", label="", commit_msg="persist data"):
    """Try ALL persistence layers. Drive -> GitHub -> HF. Never stops the process."""
    saved = False

    # Layer 1: Try Drive
    if save_to_drive(src_path, drive_dir, label):
        saved = True
    else:
        print(f"  [PERSIST] Drive failed, trying GitHub + HuggingFace fallback...")

    # Layer 2: Try GitHub
    if github_token and github_repo:
        if push_to_github(github_token, github_repo, src_path, commit_msg):
            saved = True

    # Layer 3: Try HuggingFace
    if hf_token and hf_repo:
        if push_to_hf(hf_token, hf_repo, src_path, label):
            saved = True

    if not saved:
        print(f"  [PERSIST] WARNING: All persistence layers failed! Data is only on local disk (/content/).")
        print(f"  [PERSIST] Process continues - local data is safe within this session.")
    else:
        print(f"  [PERSIST] Data persisted successfully.")

    return saved


def push_to_github(github_token, github_repo, file_path, commit_msg="update"):
    """Push a file to the GitHub checkpoint repo. Non-blocking."""
    if not github_token or not github_repo:
        return False
    try:
        github_local = Path("/content/zarx-checkpoints")
        auth_url = f"https://{github_token}@github.com/{github_repo}.git"

        if not github_local.exists():
            result = subprocess.run(
                ["git", "clone", auth_url, str(github_local)],
                capture_output=True, text=True, timeout=60
            )
            if not github_local.exists():
                github_local.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init"], cwd=str(github_local), capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", auth_url],
                             cwd=str(github_local), capture_output=True)

        # Copy file or directory
        src = Path(file_path)
        if not src.exists():
            return False

        # For large directories, only copy small files (skip huge .pt files for data)
        if src.is_dir():
            shutil.copytree(str(src), str(github_local / src.name), dirs_exist_ok=True)
        else:
            # Skip files larger than 500MB for GitHub (git push limit)
            if src.stat().st_size > 500 * 1024 * 1024:
                print(f"  [GITHUB] Skipping {src.name} (too large for GitHub: {src.stat().st_size/1e9:.1f}GB)")
                return False
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
        if result.returncode != 0:
            subprocess.run(["git", "remote", "set-url", "origin", auth_url],
                          cwd=str(github_local), capture_output=True)
            subprocess.run(["git", "push", "-u", "origin", "main"],
                          cwd=str(github_local), capture_output=True, text=True, timeout=120)
        print(f"  [GITHUB] Pushed to {github_repo}")
        return True
    except Exception as e:
        print(f"  [GITHUB] Warning: Push failed: {e}")
        return False


def push_to_hf(hf_token, hf_repo, file_path, path_in_repo=""):
    """Push a file to HuggingFace Hub. Non-blocking."""
    try:
        from huggingface_hub import HfApi
        src = Path(file_path)
        if not src.exists():
            return False
        if src.is_dir():
            # Upload directory
            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path=str(src),
                repo_id=hf_repo,
                repo_type="model",
                path_in_repo=path_in_repo if path_in_repo else src.name,
            )
            print(f"  [HF] Pushed {src.name} to {hf_repo}")
            return True
        else:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=path_in_repo if path_in_repo else src.name,
                repo_id=hf_repo,
                repo_type="model",
            )
            print(f"  [HF] Pushed {src.name} to {hf_repo}")
            return True
    except Exception as e:
        print(f"  [HF] Warning: Push failed: {e}")
        return False


def restore_from_drive(drive_path, dest_path, label=""):
    """Restore a file or directory from Google Drive if it exists."""
    global DRIVE_WORKING
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


def main():
    parser = argparse.ArgumentParser(description="ZARX-1B Day 1 Setup")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace API token")
    parser.add_argument("--wandb_key", type=str, default="", help="Weights & Biases API key")
    parser.add_argument("--hf_repo", type=str, default="Chvigo/zarx-checks", help="HuggingFace checkpoint repo ID")
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
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        if "A100" in gpu_name or "A10" in gpu_name or "RTX 30" in gpu_name or "RTX 40" in gpu_name or "L4" in gpu_name:
            run("pip install -q flash-attn --no-build-isolation", "Flash Attention 2 (Ampere+ GPU detected)", check=False)
        else:
            print(f"  Skipping Flash Attention 2 (not supported on {gpu_name})")
            print(f"  PyTorch SDPA will auto-use the best available attention backend")

    # ========== STEP 2: Detect Google Drive ==========
    step(2, "Detect Google Drive + Check Space")

    DRIVE_DIR = Path("/content/drive/MyDrive/ZARX-1B")
    if DRIVE_DIR.exists():
        try:
            stat = shutil.disk_usage(str(DRIVE_DIR))
            free_gb = stat.free / 1e9
            print(f"  Google Drive detected!")
            print(f"  Free space: {free_gb:.1f} GB")
            if free_gb < 5:
                print(f"  WARNING: Only {free_gb:.1f}GB free on Drive. Data files may not fit.")
                print(f"  Will fall back to GitHub + HuggingFace for persistence.")
            DRIVE_AVAILABLE = True
        except Exception:
            DRIVE_AVAILABLE = True
    else:
        DRIVE_DIR = Path("/content/zarx-persist")
        DRIVE_AVAILABLE = False
        print(f"  WARNING: Google Drive not found.")
        print(f"  Using fallback persist dir: {DRIVE_DIR}")

    # Create directories (may fail on full Drive, that's OK)
    try:
        for d in [DRIVE_DIR, DRIVE_DIR / "data" / "raw" / "prox-code", DRIVE_DIR / "data" / "raw" / "arc",
                  DRIVE_DIR / "data" / "processed", DRIVE_DIR / "checkpoints", DRIVE_DIR / "tokenizer", DRIVE_DIR / "configs"]:
            d.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f"  [DRIVE] Could not create directories - Drive may be full. Using GitHub fallback.")

    print(f"  Persistence strategy: Drive (try) -> GitHub (fallback) -> HF (fallback)")
    print(f"  THE PROCESS WILL NEVER STOP due to storage issues.")

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
            # PERSIST: Drive -> GitHub -> HF
            print(f"\n  [PERSIST] Saving ProX data...")
            persist(DATA_DIR / "raw" / "prox-code", DRIVE_DIR / "data" / "raw",
                   github_token=args.github_token, github_repo=args.github_repo,
                   hf_token=args.hf_token, hf_repo=args.hf_repo,
                   label="prox-code", commit_msg="add prox-code data")
    else:
        print("\n  Skipping data download (--skip_download)")
        restore_from_drive(str(DRIVE_DIR / "data" / "raw"), str(DATA_DIR / "raw"))

    # ========== STEP 7: Download ARC Data ==========
    if not args.skip_download:
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
            print(f"\n  [PERSIST] Saving ARC data...")
            persist(DATA_DIR / "raw" / "arc", DRIVE_DIR / "data" / "raw",
                   github_token=args.github_token, github_repo=args.github_repo,
                   hf_token=args.hf_token, hf_repo=args.hf_repo,
                   label="arc", commit_msg="add arc data")

    # ========== STEP 8: Generate ARC Augmentations ==========
    if not args.skip_arc_augment:
        step(8, "Generate ARC Augmented Dataset")

        arc1_path = DATA_DIR / "raw" / "arc" / "arc-agi-1" / "data" / "training"
        arc2_path = DATA_DIR / "raw" / "arc" / "arc-agi-2" / "data" / "training"
        drive_aug = DRIVE_DIR / "data" / "raw" / "arc" / "augmented_arc.jsonl"
        local_aug = DATA_DIR / "raw" / "arc" / "augmented_arc.jsonl"

        if drive_aug.exists():
            print(f"  Found existing augmented ARC on Drive! Restoring...")
            local_aug.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(drive_aug), str(local_aug))
        elif local_aug.exists():
            print(f"  Augmented ARC already exists locally.")
        elif arc1_path.exists() and arc2_path.exists():
            run(
                f"cd {BASE_DIR} && python src/arc_augment.py "
                f"--arc1_path {arc1_path} "
                f"--arc2_path {arc2_path} "
                f"--variations_per_task 500 "
                f"--output_path {local_aug} "
                f"--format code_reasoning",
                "Generate augmented ARC tasks (400K+ tasks)"
            )
            print(f"\n  [PERSIST] Saving augmented ARC data...")
            persist(local_aug, DRIVE_DIR / "data" / "raw" / "arc",
                   github_token=args.github_token, github_repo=args.github_repo,
                   hf_token=args.hf_token, hf_repo=args.hf_repo,
                   commit_msg="add augmented arc data")
        else:
            print(f"  ARC paths not found:")
            print(f"    ARC-AGI-1: {arc1_path} (exists: {arc1_path.exists()})")
            print(f"    ARC-AGI-2: {arc2_path} (exists: {arc2_path.exists()})")
    else:
        print("\n  Skipping ARC augmentation (--skip_arc_augment)")

    # ========== STEP 9: Preprocess Data ==========
    step(9, "Preprocess Data (Dedup, Filter, Merge)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    drive_processed = DRIVE_DIR / "data" / "processed"
    if drive_processed.exists() and any(drive_processed.glob("*.jsonl")):
        print(f"  Found processed data on Drive! Restoring...")
        restore_from_drive(str(drive_processed), str(PROCESSED_DIR))
    else:
        prox_dir = DATA_DIR / "raw" / "prox-code"
        if not prox_dir.exists():
            prox_dir = DATA_DIR / "raw" / "prox"

        if prox_dir.exists():
            for shard_file in sorted(prox_dir.glob("*.jsonl"))[:3]:
                output_file = PROCESSED_DIR / shard_file.name.replace('shard', 'processed')
                run(
                    f"cd {BASE_DIR} && python src/data_pipeline.py --mode process "
                    f"--input_file {shard_file} "
                    f"--output_file {output_file} "
                    f"--source prox",
                    f"Process {shard_file.name}"
                )
                print(f"\n  [PERSIST] Saving processed shard...")
                persist(output_file, DRIVE_DIR / "data" / "processed",
                       github_token=args.github_token, github_repo=args.github_repo,
                       hf_token=args.hf_token, hf_repo=args.hf_repo,
                       commit_msg=f"add processed {shard_file.name}")

        arc_file = DATA_DIR / "raw" / "arc" / "augmented_arc.jsonl"
        if arc_file.exists():
            arc_output = PROCESSED_DIR / 'arc_processed.jsonl'
            run(
                f"cd {BASE_DIR} && python src/data_pipeline.py --mode process "
                f"--input_file {arc_file} "
                f"--output_file {arc_output} "
                f"--source arc",
                "Process ARC augmented data"
            )
            print(f"\n  [PERSIST] Saving processed ARC data...")
            persist(arc_output, DRIVE_DIR / "data" / "processed",
                   github_token=args.github_token, github_repo=args.github_repo,
                   hf_token=args.hf_token, hf_repo=args.hf_repo,
                   commit_msg="add processed arc data")

        merged_file = PROCESSED_DIR / 'zarx_pretrain.jsonl'
        run(
            f"cd {BASE_DIR} && python src/data_pipeline.py --mode merge "
            f"--processed_dir {PROCESSED_DIR} "
            f"--output_file {merged_file}",
            "Merge all processed data"
        )
        print(f"\n  [PERSIST] Saving merged pretraining data...")
        persist(PROCESSED_DIR, DRIVE_DIR / "data", "processed",
               github_token=args.github_token, github_repo=args.github_repo,
               hf_token=args.hf_token, hf_repo=args.hf_repo,
               commit_msg="add merged pretrain data")

    # ========== STEP 10: Train Tokenizer ==========
    if not args.skip_tokenizer:
        step(10, "Train Custom BPE Tokenizer")

        drive_tokenizer = DRIVE_DIR / "tokenizer" / "tokenizer.json"
        local_tokenizer = BASE_DIR / "configs" / "tokenizer.json"

        if drive_tokenizer.exists():
            print(f"  Found trained tokenizer on Drive! Restoring...")
            shutil.copy2(str(drive_tokenizer), str(local_tokenizer))
        else:
            run(
                f"cd {BASE_DIR} && python scripts/train_tokenizer.py "
                f"--data_path {PROCESSED_DIR} "
                f"--output_path {local_tokenizer} "
                f"--vocab_size 32000 "
                f"--max_lines 5000000",
                "Train 32K BPE tokenizer on code + ARC corpus"
            )
            print(f"\n  [PERSIST] Saving tokenizer...")
            persist(local_tokenizer, DRIVE_DIR / "tokenizer",
                   github_token=args.github_token, github_repo=args.github_repo,
                   hf_token=args.hf_token, hf_repo=args.hf_repo,
                   commit_msg="add trained tokenizer")
    else:
        print("\n  Skipping tokenizer training (--skip_tokenizer)")
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
            hf_flags = f"--hf_repo_id {args.hf_repo} --hf_token {args.hf_token} " if args.hf_token else ""

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

    # ========== FINAL SYNC ==========
    step(12, "Final Sync - Save Everything")

    # Try all layers for final sync
    if PROCESSED_DIR.exists():
        persist(PROCESSED_DIR, DRIVE_DIR / "data", "processed",
               github_token=args.github_token, github_repo=args.github_repo,
               hf_token=args.hf_token, hf_repo=args.hf_repo,
               commit_msg="final sync - processed data")
    if (BASE_DIR / "configs" / "tokenizer.json").exists():
        persist(BASE_DIR / "configs" / "tokenizer.json", DRIVE_DIR / "tokenizer",
               github_token=args.github_token, github_repo=args.github_repo,
               hf_token=args.hf_token, hf_repo=args.hf_repo,
               commit_msg="final sync - tokenizer")
    if Path("/content/checkpoints").exists():
        persist("/content/checkpoints", DRIVE_DIR, "checkpoints",
               github_token=args.github_token, github_repo=args.github_repo,
               hf_token=args.hf_token, hf_repo=args.hf_repo,
               commit_msg="final sync - checkpoints")

    # ========== SUMMARY ==========
    elapsed = datetime.now() - start_time
    drive_status = "AVAILABLE" if DRIVE_WORKING else "FULL/DISABLED (using GitHub+HF fallback)"
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

  PERSISTENCE STATUS:
    Drive: {drive_status}
    GitHub: {args.github_repo if args.github_token else 'not configured'}
    HuggingFace: {args.hf_repo if args.hf_token else 'not configured'}
    Local: /content/ (safe within this session)

  THE PROCESS NEVER STOPS - if one storage fails, others take over.
    """)


if __name__ == "__main__":
    main()
