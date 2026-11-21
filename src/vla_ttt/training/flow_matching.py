"""Sequence flow matching, with independent noise levels for each action chunk."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FlowMatchingInputs:
    noisy_actions: Tensor
    tau: Tensor
    target_velocity: Tensor


def sample_flow_matching(
    actions: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchingInputs:
    """Use the paper's tau = .999 * (1 - Beta(1.5, 1)) independently per chunk.

    ``actions`` must already be normalized by the backbone action processor.
    The inverse CDF gives the same Beta distribution and supports a generator.
    """
    if actions.ndim != 4 or not actions.is_floating_point():
        raise ValueError("Normalized actions must be floating point [B,S,H,D].")
    uniform = torch.rand(
        actions.shape[:2], device=actions.device, dtype=torch.float32, generator=generator
    )
    tau = 0.999 * (1.0 - uniform.pow(1.0 / 1.5))
    noise = torch.randn(
        actions.shape, device=actions.device, dtype=actions.dtype, generator=generator
    )
    level = tau.to(actions.dtype)[..., None, None]
    return FlowMatchingInputs(
        noisy_actions=level * actions + (1.0 - level) * noise,
        tau=tau,
        target_velocity=actions - noise,
    )


def physical_loss_mask(batch: object) -> Tensor:
    """Supervision in physical action coordinates, before model dimension padding."""
    actions = batch.actions
    mask = batch.action_loss_mask.to(device=actions.device, dtype=torch.bool)
    mask = torch.broadcast_to(mask, actions.shape)
    action_valid = getattr(batch, "action_valid_mask", None)
    if action_valid is not None:
        mask = mask & action_valid.to(device=actions.device, dtype=torch.bool)
    valid = batch.valid_mask.to(device=actions.device, dtype=torch.bool)
    return mask & valid[..., None, None]


def sequence_denominators(batch: object) -> Tensor:
    """Count supervised elements over each entire logical sequence, not a segment."""
    return physical_loss_mask(batch).sum(dim=(1, 2, 3), dtype=torch.float32)


def normalized_loss_mask(
    physical_mask: Tensor,
    normalized_actions: Tensor,
    normalization_valid_mask: Tensor,
) -> Tensor:
    """Pad physical dimensions, then intersect with the backbone's valid dimensions.

    RoboMME-Plus uses eight physical action dimensions in the first eight slots
    of the GR00T action representation. Other mappings need an explicit adapter.
    """
    if physical_mask.ndim != 4 or normalized_actions.ndim != 4:
        raise ValueError("Action tensors and masks must have shape [B,S,H,D].")
    if physical_mask.shape[:2] != normalized_actions.shape[:2]:
        raise ValueError("Normalization must preserve batch and sequence axes.")
    extra = normalized_actions.shape[-1] - physical_mask.shape[-1]
    extra_horizon = normalized_actions.shape[-2] - physical_mask.shape[-2]
    if extra < 0 or extra_horizon < 0:
        raise ValueError("Normalization cannot shorten the physical action horizon or dimension.")
    padded = F.pad(
        physical_mask.to(normalized_actions.device, dtype=torch.bool), (0, extra, 0, extra_horizon)
    )
    valid = normalization_valid_mask.to(device=normalized_actions.device, dtype=torch.bool)
    try:
        mask = padded & torch.broadcast_to(valid, normalized_actions.shape)
    except RuntimeError as exc:
        raise ValueError("Normalization mask is not broadcastable to normalized actions.") from exc
    # The logical sequence denominator is computed before decoding video. Reject
    # adapters that silently remove supervision and would invalidate that count.
    if not torch.equal(mask.sum((1, 2, 3)), padded.sum((1, 2, 3))):
        raise ValueError("Action normalization removed supervised physical dimensions.")
    return mask


def masked_sequence_loss_sum(
    velocity: Tensor,
    target_velocity: Tensor,
    mask: Tensor,
    full_sequence_denominators: Tensor,
) -> Tensor:
    """Sum sequence-normalized losses for this segment, before global batch mean.

    Adding these values across segments gives a sum of per-sequence means. The
    trainer divides by the global number of supervised sequences, including all
    ranks. A rank with no supervision contributes zero to that mean.
    """
    if velocity.shape != target_velocity.shape or mask.shape != velocity.shape:
        raise ValueError("Velocity, target, and loss mask must have identical shapes.")
    if full_sequence_denominators.shape != velocity.shape[:1]:
        raise ValueError("Expected one full-sequence denominator per batch sample.")
    residual = torch.where(
        mask, velocity.float() - target_velocity.float(), torch.zeros((), device=velocity.device)
    )
    numerator = residual.square().sum(dim=(1, 2, 3))
    denominator = full_sequence_denominators.to(device=velocity.device, dtype=torch.float32)
    return (numerator / denominator.clamp_min(1.0)).sum()
