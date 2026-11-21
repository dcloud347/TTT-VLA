"""Differentiable key/value binding in a two-layer GeLU fast model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fast_weights import FastWeights


class TTTMLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, inner_lr: float = 0.1):
        super().__init__()
        if dimension <= 0 or hidden_dimension <= 0 or inner_lr <= 0:
            raise ValueError("Fast model dimensions and inner_lr must be positive")
        self.w1 = nn.Parameter(torch.empty(dimension, hidden_dimension))
        self.b1 = nn.Parameter(torch.zeros(1, hidden_dimension))
        self.w2 = nn.Parameter(torch.empty(hidden_dimension, dimension))
        self.b2 = nn.Parameter(torch.zeros(1, dimension))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        self.log_lr = nn.Parameter(torch.zeros(()))
        self.base_lr = float(inner_lr)

    def init_state(self, batch_size: int, *, training: bool, device=None) -> FastWeights:
        parameters = (self.w1, self.b1, self.w2, self.b2)
        values = []
        for parameter in parameters:
            source = parameter if training else parameter.detach()
            value = source.to(device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, *parameter.shape).clone()
            # Detached TBPTT states and inference memories remain differentiable locally.
            if not value.requires_grad:
                value.requires_grad_(True)
            values.append(value)
        return FastWeights(*values)

    @staticmethod
    def apply(x: Tensor, state: FastWeights) -> Tensor:
        return torch.bmm(F.gelu(torch.bmm(x, state.w1) + state.b1), state.w2) + state.b2

    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, state: FastWeights,
        *, token_mask: Tensor, update_mask: Tensor, create_graph: bool,
    ) -> tuple[Tensor, FastWeights]:
        if torch.is_inference_mode_enabled():
            raise RuntimeError("TTT requires local autograd: use torch.no_grad(), not torch.inference_mode()")
        if not all(x.requires_grad for x in state.tensors()):
            state = state.detach(requires_grad=True)
        # Inference callers use no_grad for QKV and the slow backbone. Only the
        # small fast model needs autograd to compute its update.
        with torch.enable_grad(), torch.autocast(device_type=keys.device.type, enabled=False):
            keys, values = keys.float(), values.float()
            prediction = self.apply(keys, state)
            weights = token_mask.to(prediction.dtype) * update_mask[:, None].to(prediction.dtype)
            count = weights.sum(dim=-1).clamp_min(1) * prediction.shape[-1]
            per_sample = ((prediction - values).square() * weights[..., None]).sum(dim=(-2, -1)) / count
            gradients = torch.autograd.grad(per_sample.sum(), state.tensors(), create_graph=create_graph)
            lr = self.base_lr * F.softplus(self.log_lr.float()) / math.log(2.0)
            if not create_graph:
                lr = lr.detach()
            candidate = FastWeights(*(w - lr * g for w, g in zip(state.tensors(), gradients)))
            next_state = state.select(update_mask, candidate)
            if not create_graph:
                next_state = next_state.detach(requires_grad=True)
        with torch.set_grad_enabled(create_graph), torch.autocast(device_type=keys.device.type, enabled=False):
            output = self.apply(queries.float(), next_state)
        return output, next_state

