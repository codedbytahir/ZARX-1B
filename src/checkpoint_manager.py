"""
ZARX-1B Bulletproof Checkpoint Manager
Triple-redundancy system: Local + Google Drive + HuggingFace Hub

Layer 1: Local disk     — every 100 steps  (fast, survives within session)
Layer 2: Google Drive   — every 500 steps  (survives Colab restarts)
Layer 3: HuggingFace    — every 1000 steps (survives EVERYTHING, cross-platform)
"""

import torch
import os
import time
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class CheckpointManager:
    """Triple-redundancy checkpoint system for ZARX-1B training."""

    def __init__(
        self,
        local_dir: str = "/content/checkpoints",
        drive_dir: str = "/content/drive/MyDrive/ZARX-1B",
        hf_repo_id: str = None,
        hf_token: str = None,
        keep_local: int = 3,
        keep_drive: int = 3,
        keep_hf: int = 5,
        save_every_local: int = 100,
        save_every_drive: int = 500,
        save_every_hf: int = 1000,
    ):
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.hf_api = None

        self.keep_local = keep_local
        self.keep_drive = keep_drive
        self.keep_hf = keep_hf

        self.save_every_local = save_every_local
        self.save_every_drive = save_every_drive
        self.save_every_hf = save_every_hf

        # Create directories
        self.local_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.drive_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            print("[CHECKPOINT] Warning: Google Drive not mounted yet. Drive saves will be skipped.")

        # Initialize HF
        if HF_AVAILABLE and hf_repo_id and hf_token:
            try:
                self.hf_api = HfApi(token=hf_token)
                # Ensure repo exists
                try:
                    self.hf_api.repo_info(repo_id=hf_repo_id, repo_type="model")
                except Exception:
                    self.hf_api.create_repo(repo_id=hf_repo_id, repo_type="model", private=True)
                    print(f"[CHECKPOINT] Created HF repo: {hf_repo_id}")
            except Exception as e:
                print(f"[CHECKPOINT] Warning: HF initialization failed: {e}")
                self.hf_api = None

    def should_save(self, step: int) -> list:
        """Determine which layers need saving at this step."""
        layers = []
        if step % self.save_every_local == 0 and step > 0:
            layers.append("local")
        if step % self.save_every_drive == 0 and step > 0:
            layers.append("drive")
        if step % self.save_every_hf == 0 and step > 0:
            layers.append("hf")
        return layers

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        total_tokens: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        extra_state: Optional[Dict] = None,
    ) -> None:
        """Save checkpoint to all required layers."""

        layers = self.should_save(step)
        if not layers:
            return

        print(f"[CHECKPOINT] Saving step {step} to: {', '.join(layers)}")

        # Build checkpoint dict
        checkpoint = {
            "step": step,
            "total_tokens": total_tokens,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.utcnow().isoformat(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "extra_state": extra_state or {},
        }

        filename = f"checkpoint-step-{step}.pt"
        local_path = self.local_dir / filename
        latest_path = self.local_dir / "checkpoint-latest.pt"

        # === LAYER 1: LOCAL (always save here first) ===
        print(f"[CHECKPOINT] Writing to local disk...")
        torch.save(checkpoint, local_path)
        torch.save(checkpoint, latest_path)

        # Save lightweight metadata
        meta = {
            "latest_step": step,
            "total_tokens": total_tokens,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": checkpoint["timestamp"],
        }
        meta_path = self.local_dir / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # === LAYER 2: GOOGLE DRIVE ===
        if "drive" in layers:
            print(f"[CHECKPOINT] Copying to Google Drive...")
            try:
                drive_path = self.drive_dir / filename
                drive_latest = self.drive_dir / "checkpoint-latest.pt"
                drive_meta = self.drive_dir / "training_metadata.json"

                shutil.copy2(local_path, drive_path)
                shutil.copy2(local_path, drive_latest)
                shutil.copy2(meta_path, drive_meta)

                self._cleanup_dir(self.drive_dir, self.keep_drive)
                print(f"[CHECKPOINT] Drive save complete.")
            except Exception as e:
                print(f"[CHECKPOINT] Warning: Drive save failed: {e}")

        # === LAYER 3: HUGGINGFACE HUB ===
        if "hf" in layers and self.hf_api:
            print(f"[CHECKPOINT] Pushing to HuggingFace Hub...")
            try:
                self.hf_api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=filename,
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                )
                self.hf_api.upload_file(
                    path_or_fileobj=str(latest_path),
                    path_in_repo="checkpoint-latest.pt",
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                )
                self.hf_api.upload_file(
                    path_or_fileobj=str(meta_path),
                    path_in_repo="training_metadata.json",
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                )
                self._cleanup_hf(self.keep_hf)
                print(f"[CHECKPOINT] HF Hub push complete!")
            except Exception as e:
                print(f"[CHECKPOINT] Warning: HF push failed: {e}")
                print(f"[CHECKPOINT] Local + Drive copies are safe. Will retry next cycle.")

        # Cleanup local
        self._cleanup_dir(self.local_dir, self.keep_local)
        print(f"[CHECKPOINT] Step {step} saved successfully.")

    def load_latest(self) -> Optional[Dict]:
        """Load the latest checkpoint, trying all three layers."""

        # Try Layer 1: Local
        ckpt = self._try_load_from_dir(self.local_dir)
        if ckpt:
            print(f"[RESUME] Loaded from LOCAL disk (step {ckpt['step']})")
            return ckpt

        # Try Layer 2: Google Drive
        ckpt = self._try_load_from_dir(self.drive_dir)
        if ckpt:
            print(f"[RESUME] Loaded from GOOGLE DRIVE (step {ckpt['step']})")
            return ckpt

        # Try Layer 3: HuggingFace Hub
        if self.hf_api and self.hf_repo_id:
            ckpt = self._try_load_hf()
            if ckpt:
                print(f"[RESUME] Loaded from HUGGINGFACE HUB (step {ckpt['step']})")
                return ckpt

        print("[RESUME] No checkpoint found. Starting from scratch.")
        return None

    def _try_load_from_dir(self, directory: Path) -> Optional[Dict]:
        """Try loading from a directory, with fallback to older checkpoints."""
        latest = directory / "checkpoint-latest.pt"
        if latest.exists():
            try:
                return torch.load(latest, map_location="cpu")
            except Exception as e:
                print(f"[RESUME] Latest checkpoint corrupted: {e}")

        # Fallback: find newest step checkpoint
        return self._find_latest_in_dir(directory)

    def _find_latest_in_dir(self, directory: Path) -> Optional[Dict]:
        """Find the latest checkpoint by step number."""
        checkpoints = []
        for f in directory.glob("checkpoint-step-*.pt"):
            try:
                step = int(f.name.split("-")[-1].replace(".pt", ""))
                checkpoints.append((step, f))
            except ValueError:
                continue

        if not checkpoints:
            return None

        checkpoints.sort(reverse=True)
        for step, path in checkpoints:
            try:
                return torch.load(path, map_location="cpu")
            except Exception:
                continue
        return None

    def _try_load_hf(self) -> Optional[Dict]:
        """Try loading from HuggingFace Hub."""
        if not HF_AVAILABLE:
            return None

        try:
            # Download metadata first
            meta_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="training_metadata.json",
                repo_type="model",
                token=self.hf_token,
            )
            with open(meta_path) as f:
                meta = json.load(f)

            latest_step = meta["latest_step"]
            print(f"[RESUME] HF has checkpoint at step {latest_step}. Downloading...")

            ckpt_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"checkpoint-step-{latest_step}.pt",
                repo_type="model",
                token=self.hf_token,
            )
            return torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[RESUME] HF metadata download failed: {e}")

        # Fallback: try generic latest pointer
        try:
            ckpt_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="checkpoint-latest.pt",
                repo_type="model",
                token=self.hf_token,
            )
            return torch.load(ckpt_path, map_location="cpu")
        except Exception:
            return None

    def _cleanup_dir(self, directory: Path, keep: int):
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoints = []
        for f in directory.glob("checkpoint-step-*.pt"):
            try:
                step = int(f.name.split("-")[-1].replace(".pt", ""))
                checkpoints.append((step, f))
            except ValueError:
                continue

        checkpoints.sort(reverse=True)
        for step, path in checkpoints[keep:]:
            try:
                path.unlink()
            except Exception:
                pass

    def _cleanup_hf(self, keep: int):
        """Remove old checkpoints from HuggingFace Hub."""
        if not self.hf_api:
            return
        try:
            files = self.hf_api.list_repo_files(
                repo_id=self.hf_repo_id, repo_type="model"
            )
            checkpoints = []
            for f in files:
                if f.startswith("checkpoint-step-") and f.endswith(".pt"):
                    try:
                        step = int(f.split("-")[-1].replace(".pt", ""))
                        checkpoints.append((step, f))
                    except ValueError:
                        continue

            checkpoints.sort(reverse=True)
            for step, filename in checkpoints[keep:]:
                try:
                    self.hf_api.delete_file(
                        path_in_repo=filename,
                        repo_id=self.hf_repo_id,
                        repo_type="model",
                    )
                    print(f"[CLEANUP] Removed HF: {filename}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[CLEANUP] HF cleanup failed: {e}")

    def emergency_recover(self) -> Optional[Dict]:
        """Scan ALL three layers and find the most advanced checkpoint."""
        print("[EMERGENCY] Scanning all storage layers for checkpoints...")
        candidates = []

        # Scan local
        for f in self.local_dir.glob("checkpoint-step-*.pt"):
            try:
                step = int(f.name.split("-")[-1].replace(".pt", ""))
                candidates.append(("local", step, f))
            except ValueError:
                pass

        # Scan Drive
        for f in self.drive_dir.glob("checkpoint-step-*.pt"):
            try:
                step = int(f.name.split("-")[-1].replace(".pt", ""))
                candidates.append(("drive", step, f))
            except ValueError:
                pass

        # Scan HF
        if self.hf_api and self.hf_repo_id:
            try:
                files = self.hf_api.list_repo_files(
                    repo_id=self.hf_repo_id, repo_type="model"
                )
                for f in files:
                    if f.startswith("checkpoint-step-") and f.endswith(".pt"):
                        try:
                            step = int(f.split("-")[-1].replace(".pt", ""))
                            candidates.append(("hf", step, f))
                        except ValueError:
                            pass
            except Exception:
                pass

        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"[EMERGENCY] Found {len(candidates)} checkpoints:")
        for source, step, path in candidates[:10]:
            print(f"  [{source}] Step {step}: {path}")

        if not candidates:
            return None

        for source, step, path in candidates:
            try:
                if source == "hf" and HF_AVAILABLE:
                    path = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename=path,
                        repo_type="model",
                        token=self.hf_token,
                    )
                ckpt = torch.load(path, map_location="cpu")
                print(f"[EMERGENCY] Recovered step {step} from {source}!")
                print(f"  Tokens: {ckpt['total_tokens']/1e9:.2f}B, Loss: {ckpt['train_loss']:.4f}")
                return ckpt
            except Exception:
                continue

        print("[EMERGENCY] All checkpoints corrupted!")
        return None
