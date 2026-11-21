"""Explicit per-session memory. No runtime state is stored in module buffers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FastWeights:
    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.w1, self.b1, self.w2, self.b2

    def detach(self, requires_grad: bool = True) -> FastWeights:
        return FastWeights(*(x.detach().requires_grad_(requires_grad) for x in self.tensors()))

    def select(self, mask: Tensor, replacement: FastWeights) -> FastWeights:
        """Replace only selected batch members, preserving the computation graph."""
        return FastWeights(*(
            torch.where(mask.reshape(-1, *([1] * (old.ndim - 1))), new, old)
            for old, new in zip(self.tensors(), replacement.tensors())
        ))


@dataclass(frozen=True)
class MemoryState:
    layers: tuple[FastWeights, ...]
    positions: Tensor
    commits: Tensor
    session_ids: tuple[str, ...] = ()

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])

    def detach(self, requires_grad: bool = True) -> MemoryState:
        return MemoryState(
            tuple(x.detach(requires_grad) for x in self.layers),
            self.positions.detach(), self.commits.detach(), self.session_ids,
        )

