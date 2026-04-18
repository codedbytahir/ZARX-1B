"""Tests for ZARX-1B model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ZARXModel, ZARXConfig


def test_model_creation():
    """Test that model can be created with default config."""
    config = ZARXConfig()
    model = ZARXModel(config)
    model.tie_weights()
    params = model.count_parameters()
    print(f"Parameters: {params['total_billion']:.3f}B")
    assert 0.9e9 < params['total'] < 1.1e9, f"Expected ~1B params, got {params['total']}"
    print("PASS: Model creation")


def test_forward_pass():
    """Test forward pass with random input."""
    import torch
    config = ZARXConfig()
    model = ZARXModel(config)
    model.tie_weights()

    x = torch.randint(0, config.vocab_size, (1, 64))
    out = model(x)
    assert out["logits"].shape == (1, 64, config.vocab_size), f"Wrong output shape: {out['logits'].shape}"
    print("PASS: Forward pass")


def test_forward_with_labels():
    """Test forward pass with labels (computes loss)."""
    import torch
    config = ZARXConfig()
    model = ZARXModel(config)
    model.tie_weights()

    x = torch.randint(0, config.vocab_size, (1, 64))
    labels = x.clone()
    out = model(x, labels=labels)
    assert out["loss"] is not None, "Loss should be computed"
    assert out["loss"].item() > 0, "Loss should be positive"
    print(f"PASS: Forward with labels (loss={out['loss'].item():.4f})")


def test_backward_pass():
    """Test backward pass (gradient flow)."""
    import torch
    config = ZARXConfig()
    model = ZARXModel(config)
    model.tie_weights()

    x = torch.randint(0, config.vocab_size, (1, 64))
    labels = x.clone()
    out = model(x, labels=labels)
    out["loss"].backward()

    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"PASS: Backward pass ({has_grad}/{total_params} params have gradients)")


def test_gradient_checkpointing():
    """Test gradient checkpointing."""
    import torch
    config = ZARXConfig()
    model = ZARXModel(config)
    model.tie_weights()
    model.enable_gradient_checkpointing()
    model.train()

    x = torch.randint(0, config.vocab_size, (1, 64))
    labels = x.clone()
    out = model(x, labels=labels)
    out["loss"].backward()
    print("PASS: Gradient checkpointing")


def test_config_save_load():
    """Test config save and load."""
    import tempfile
    config = ZARXConfig()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        config.to_json(f.name)
        loaded = ZARXConfig.from_json(f.name)
        assert loaded.vocab_size == config.vocab_size
        assert loaded.hidden_size == config.hidden_size
        print("PASS: Config save/load")


if __name__ == "__main__":
    print("Running ZARX-1B Model Tests\n" + "=" * 40)
    test_model_creation()
    test_forward_pass()
    test_forward_with_labels()
    test_backward_pass()
    test_gradient_checkpointing()
    test_config_save_load()
    print("\n" + "=" * 40)
    print("All tests passed!")
