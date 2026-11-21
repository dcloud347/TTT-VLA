"""Learned projections, rotary positions and gated fast-weight residual."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from .fast_weights import FastWeights
from .ttt_mlp import TTTMLP


@dataclass(frozen=True)
class TTTConfig:
    memory_dim: int = 1024
    fast_hidden_dim: int = 1536
    num_register_tokens: int = 16
    gate_init: float = 0.001
    inner_lr: float = 0.1
    rope_theta: float = 10000.0
    normalize_input: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> TTTConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(config) - names
        if unknown:
            raise ValueError(f"Unknown TTT settings: {sorted(unknown)}")
        value = cls(**config)
        if value.memory_dim <= 0 or value.memory_dim % 2 or value.fast_hidden_dim <= 0:
            raise ValueError("memory_dim must be positive and even; fast_hidden_dim must be positive")
        if value.num_register_tokens <= 0 or value.rope_theta <= 0:
            raise ValueError("register count and RoPE theta must be positive")
        return value


class TTTBlock(nn.Module):
    def __init__(self, hidden_size: int, config: TTTConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(hidden_size) if config.normalize_input else nn.Identity()
        self.q = nn.Linear(hidden_size, config.memory_dim, bias=False)
        self.k = nn.Linear(hidden_size, config.memory_dim, bias=False)
        self.v = nn.Linear(hidden_size, config.memory_dim, bias=False)
        self.out = nn.Linear(config.memory_dim, hidden_size, bias=False)
        self.fast = TTTMLP(config.memory_dim, config.fast_hidden_dim, config.inner_lr)
        self.alpha = nn.Parameter(torch.full((hidden_size,), config.gate_init))
        frequency = 1.0 / config.rope_theta ** (torch.arange(0, config.memory_dim, 2).float() / config.memory_dim)
        self.register_buffer("rope_frequency", frequency, persistent=False)

    def _rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        indices = positions[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]
        angles = indices.float()[..., None] * self.rope_frequency.float()
        cosine, sine = angles.cos(), angles.sin()
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(
        self, hidden: Tensor, state: FastWeights, positions: Tensor,
        *, token_mask: Tensor, update_mask: Tensor, create_graph: bool,
    ) -> tuple[Tensor, FastWeights]:
        # Keep the memory branch in FP32 even when the backbone is BF16.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            normalized = self.norm(hidden.float())
            q = self._rotate(self.q(normalized), positions)
            k = self._rotate(self.k(normalized), positions)
            v = self.v(normalized)
            output, next_state = self.fast(
                q, k, v, state, token_mask=token_mask, update_mask=update_mask, create_graph=create_graph,
            )
            residual = self.out(output) * torch.tanh(self.alpha)
            residual = residual * token_mask[..., None].to(residual.dtype)
        return hidden + residual.to(hidden.dtype), next_state

