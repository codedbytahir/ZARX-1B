# ZARX-1B

A 1B parameter decoder-only transformer trained from scratch for **code generation** and **ARC-AGI abstract reasoning**.

## Architecture

| Component | Value |
|-----------|-------|
| Parameters | ~1B |
| Hidden Size | 2048 |
| Layers | 24 |
| Attention Heads | 16 |
| KV Heads (GQA) | 4 |
| Context Length | 2048 |
| Activation | SwiGLU |
| Norm | RMSNorm |
| Position Encoding | RoPE |
| Vocab Size | 32,000 (custom BPE) |

## Features

- **LLaMA-style** architecture with modern optimizations
- **Grouped Query Attention (GQA)** for efficient inference
- **SwiGLU** activation (better than GELU)
- **Flash Attention 2** compatible
- **Gradient checkpointing** for memory efficiency
- **Triple-redundancy checkpoint system** (Local + Drive + HuggingFace Hub)
- **ARC-AGI augmentation pipeline** with code-as-reasoning format
- **8-bit AdamW** optimizer for VRAM savings
- Designed for **Google Colab + Kaggle free tier** (single T4 GPU)

## Training Data

| Source | Type | Proportion |
|--------|------|-----------|
| ProX Refined Code | Code (quality-refined) | 80% |
| ARC-AGI (augmented) | Abstract reasoning | 20% |

## Quick Start

### 1. Setup on Google Colab

Open `scripts/colab_train.ipynb` in Google Colab with a T4 GPU runtime.

### 2. Or run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py --download_all

# Generate ARC augmented data
python src/arc_augment.py \
    --arc1_path data/raw/arc/arc-agi-1/training \
    --arc2_path data/raw/arc/arc-agi-2/data/training \
    --output_path data/processed/arc_augmented.jsonl

# Train tokenizer
python scripts/train_tokenizer.py --data_path data/processed

# Start training
python src/train.py \
    --model_config configs/model_config.json \
    --tokenizer_path configs/tokenizer.json \
    --data_path data/processed
```

## Project Structure

```
ZARX-1B/
├── configs/          # Model, tokenizer, training configs
├── data/             # Raw and processed data
├── src/              # Core source code
│   ├── model.py              # Transformer architecture
│   ├── train.py              # Training loop
│   ├── checkpoint_manager.py # Triple-redundancy checkpoints
│   ├── arc_augment.py        # ARC augmentation pipeline
│   ├── data_pipeline.py      # Data preprocessing
│   └── eval.py               # Evaluation suite
├── scripts/          # Notebooks and utility scripts
└── tests/            # Unit tests
```

## Checkpoint System

Three-layer redundancy ensures no training data is ever lost:

1. **Local disk** — every 100 steps (fast, in-session)
2. **Google Drive** — every 500 steps (survives Colab restarts)
3. **HuggingFace Hub** — every 1000 steps (cross-platform, survives everything)

Maximum possible data loss: ~8 minutes of training.

## License

MIT
