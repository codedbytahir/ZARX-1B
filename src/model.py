"""
ZARX-1B Model Architecture
LLaMA-style decoder-only transformer with modern optimizations.

Architecture: ~1B parameters
- 24 layers, 2048 hidden size, 16 attention heads
- Grouped Query Attention (4 KV heads)
- SwiGLU activation, RMSNorm, RoPE positional encoding
- Tied embeddings (input/output shared)
- Flash Attention 2 compatible
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ZARXConfig:
    """Configuration for ZARX-1B model."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA
    intermediate_size: int = 5632  # SwiGLU: ~2.75x hidden
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            return cls(**json.load(f))

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA)."""

    def __init__(self, config: ZARXConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (before KV expansion to save memory)
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for GQA (after RoPE to avoid computing RoPE on duplicated tensors)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Flash Attention via PyTorch SDPA
        # Use is_causal=True when no custom mask — SDPA handles it efficiently
        # This avoids creating a huge seq_len x seq_len attention matrix
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True if attention_mask is None else False,
        )

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP (better than standard GELU MLP)."""

    def __init__(self, config: ZARXConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, config: ZARXConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ZARXModel(nn.Module):
    """
    ZARX-1B: A 1B parameter decoder-only transformer for code + ARC reasoning.

    Features:
    - LLaMA-style architecture with SwiGLU, RMSNorm, RoPE
    - Grouped Query Attention (GQA) for efficient inference
    - Tied input/output embeddings
    - Flash Attention 2 compatible
    - Gradient checkpointing support
    """

    def __init__(self, config: ZARXConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Output projection tied to embedding (set in tie_weights)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def tie_weights(self):
        """Tie output projection to input embedding."""
        self.lm_head = self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_len = input_ids.shape

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Causal mask — ONLY build explicit mask for padding
        # When there's no padding mask, we skip the huge (seq_len x seq_len) matrix
        # and let SDPA handle causality via is_causal=True (much more memory efficient)
        if attention_mask is not None:
            # Check if there's actual padding (not all 1s)
            has_padding = (attention_mask == 0).any()
            if has_padding:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                    diagonal=1,
                )
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, None, :]
                causal_mask = causal_mask + attention_mask
            else:
                # All tokens are valid — no need for explicit mask
                causal_mask = None
        else:
            causal_mask = None

        # Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, causal_mask)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head (tied)
        logits = F.linear(hidden_states, self.embed_tokens.weight)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = self.embed_tokens.weight.numel()
        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding,
            "non_embedding": total - embedding,
            "total_billion": total / 1e9,
        }

    @classmethod
    def from_config(cls, config_path: str) -> "ZARXModel":
        """Load model from config JSON file."""
        config = ZARXConfig.from_json(config_path)
        model = cls(config)
        model.tie_weights()
        return model

    def save_pretrained(self, output_dir: str):
        """Save model and config."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_json(str(output_dir / "model_config.json"))

        # Save weights
        torch.save(self.state_dict(), str(output_dir / "model.pt"))

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ZARXModel":
        """Load model from directory."""
        model_dir = Path(model_dir)
        config = ZARXConfig.from_json(str(model_dir / "model_config.json"))
        model = cls(config)
        model.tie_weights()

        state_dict = torch.load(str(model_dir / "model.pt"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model


def build_model(config_path: str = None) -> ZARXModel:
    """Build ZARX-1B model from config or defaults."""
    if config_path:
        model = ZARXModel.from_config(config_path)
    else:
        config = ZARXConfig()
        model = ZARXModel(config)
        model.tie_weights()

    params = model.count_parameters()
    print(f"ZARX-1B Model Built:")
    print(f"  Total parameters: {params['total_billion']:.3f}B")
    print(f"  Embedding params: {params['embedding'] / 1e6:.1f}M")
    print(f"  Non-embedding:    {params['non_embedding'] / 1e6:.1f}M")
    print(f"  Model size (bf16): {params['total'] * 2 / 1e9:.2f} GB")

    return model


if __name__ == "__main__":
    model = build_model()
    print("\nTesting forward pass...")
    x = torch.randint(0, 32000, (1, 128))
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out['logits'].shape}")
    if out['loss'] is not None:
        print(f"Loss: {out['loss'].item():.4f}")
    print("Forward pass successful!")
