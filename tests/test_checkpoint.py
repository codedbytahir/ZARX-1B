"""Tests for ZARX-1B checkpoint manager."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint_manager import CheckpointManager


def test_local_save_load():
    """Test local save and load."""
    import torch
    import torch.nn as nn

    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = Path(tmpdir) / "checkpoints"

        mgr = CheckpointManager(
            local_dir=str(local_dir),
            drive_dir=str(Path(tmpdir) / "drive"),
            save_every_local=100,
            save_every_drive=999999,
            save_every_hf=999999,
        )

        # Create a simple model
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Save
        mgr.save(model, optimizer, None, step=100, total_tokens=6553600, train_loss=5.0)

        # Load
        ckpt = mgr.load_latest()
        assert ckpt is not None, "Checkpoint should exist"
        assert ckpt["step"] == 100
        assert ckpt["total_tokens"] == 6553600
        assert ckpt["train_loss"] == 5.0

        print("PASS: Local save/load")


def test_should_save():
    """Test save schedule logic."""
    mgr = CheckpointManager(
        local_dir="/tmp/test_ckpt",
        drive_dir="/tmp/test_drive",
        save_every_local=100,
        save_every_drive=500,
        save_every_hf=1000,
    )

    assert mgr.should_save(0) == []
    assert mgr.should_save(50) == []
    assert mgr.should_save(100) == ["local"]
    assert mgr.should_save(500) == ["local", "drive"]
    assert mgr.should_save(1000) == ["local", "drive", "hf"]
    assert mgr.should_save(2000) == ["local", "drive", "hf"]
    assert mgr.should_save(600) == ["local", "drive"]

    print("PASS: Should save schedule")


def test_no_checkpoint():
    """Test loading when no checkpoint exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            local_dir=str(Path(tmpdir) / "empty"),
            drive_dir=str(Path(tmpdir) / "empty_drive"),
        )
        ckpt = mgr.load_latest()
        assert ckpt is None, "Should return None when no checkpoint"
        print("PASS: No checkpoint returns None")


if __name__ == "__main__":
    print("Running ZARX-1B Checkpoint Tests\n" + "=" * 40)
    test_local_save_load()
    test_should_save()
    test_no_checkpoint()
    print("\n" + "=" * 40)
    print("All checkpoint tests passed!")
