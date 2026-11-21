"""CPU trajectory batches with image decoding deferred until a TBPTT slice."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Iterator, Sequence

import torch


Observation = dict[str, Any]


@dataclass(frozen=True)
class LazyObservationGrid:
    """A B-by-S reference grid; reading action tensors never decodes images."""

    rows: Sequence[Sequence[Any | None]]
    loader: Callable[[Sequence[Sequence[Any | None]]], list[list[Observation | None]]]

    def materialize(self) -> list[list[Observation | None]]:
        return self.loader(self.rows)

    def slice_time(self, start: int, end: int) -> "LazyObservationGrid":
        return LazyObservationGrid([row[start:end] for row in self.rows], self.loader)

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class TrajectoryBatch:
    """Actions are physical absolute targets, not GR00T-normalized targets.

    ``loss_normalizer`` is computed for the complete logical sequence and is
    deliberately preserved by ``slice_time``. Images and named proprioception
    stay on the CPU for the backbone processor.
    """

    actions: torch.Tensor
    action_loss_mask: torch.Tensor
    action_valid_mask: torch.Tensor
    valid_mask: torch.Tensor
    state_update_mask: torch.Tensor
    reset_mask: torch.Tensor
    positions: torch.Tensor
    observations: LazyObservationGrid | list[list[Observation | None]]
    loss_normalizer: torch.Tensor
    sequence_supervised_mask: torch.Tensor
    episode_ids: tuple[str, ...]
    is_video_demo: torch.Tensor | None = None
    source_labels: tuple[tuple[str, ...], ...] | None = None
    metadata: tuple[dict[str, Any], ...] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.actions.shape[1])

    def materialize_observations(self) -> "TrajectoryBatch":
        if isinstance(self.observations, LazyObservationGrid):
            return replace(self, observations=self.observations.materialize())
        return self

    def slice_time(self, start: int, end: int) -> "TrajectoryBatch":
        if not 0 <= start < end <= self.sequence_length:
            raise ValueError(f"Invalid time slice [{start}, {end}) for S={self.sequence_length}")
        time_fields = (
            "actions", "action_loss_mask", "action_valid_mask", "valid_mask",
            "state_update_mask", "reset_mask", "positions", "is_video_demo",
        )
        updates = {
            name: getattr(self, name)[:, start:end]
            for name in time_fields if getattr(self, name) is not None
        }
        if isinstance(self.observations, LazyObservationGrid):
            updates["observations"] = self.observations.slice_time(start, end).materialize()
        else:
            updates["observations"] = [row[start:end] for row in self.observations]
        if self.source_labels is not None:
            updates["source_labels"] = tuple(row[start:end] for row in self.source_labels)
        return replace(self, **updates)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "TrajectoryBatch":
        updates = {
            field.name: value.to(device=device, non_blocking=non_blocking)
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), torch.Tensor)
        }
        return replace(self, **updates)

    def iter_segments(self, segment_steps: int) -> Iterator["TrajectoryBatch"]:
        if segment_steps < 1:
            raise ValueError("segment_steps must be positive")
        for start in range(0, self.sequence_length, segment_steps):
            yield self.slice_time(start, min(start + segment_steps, self.sequence_length))


# Public alias retained for callers that already name a TBPTT slice SegmentBatch.
SegmentBatch = TrajectoryBatch
