"""Segmentation helpers; fast-weight detachment stays with the policy's state type."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

from .flow_matching import physical_loss_mask


def segment_ranges(batch: Any, segment_steps: int) -> list[tuple[int, int]]:
    """Choose bounded, nonoverlapping segments that preserve nearby context credit.

    A cut directly between an unsupervised state-writing frame and a supervised
    frame would prevent the latter's loss from teaching the former's write. Move
    such a cut earlier, considering the union of transitions across batch members.
    This protects video-to-execution and failure-to-correction transitions, as
    well as observation-only frames preceding ordinary supervised action chunks.

    For one sample and a limit of at least two frames, an immediately preceding
    context frame and its supervision share a segment. This cannot be guaranteed
    for all samples when their protected boundaries densely conflict, or when the
    limit is one frame. In those cases the maximum length wins and a warning is
    emitted. Earlier demonstration frames remain detached burn-in: this does not
    preserve gradients through an entire long video or across a long gap between
    a video and the first supervised robot action.
    """
    if segment_steps <= 0:
        raise ValueError("segment_steps must be positive.")
    length = int(batch.sequence_length)
    if length == 0:
        return []
    valid = batch.valid_mask.bool()
    supervised = physical_loss_mask(batch).any(dim=(-1, -2))
    context = valid & batch.state_update_mask.bool() & ~supervised
    # A reset begins a new session, so the preceding context must not receive
    # this session's credit even if both happen to occupy one physical segment.
    protected = (
        context[:, :-1] & supervised[:, 1:] & ~batch.reset_mask[:, 1:].bool()
    ).any(dim=0)
    protected_indices = {
        index + 1 for index in protected.nonzero(as_tuple=False).flatten().cpu().tolist()
    }
    ranges: list[tuple[int, int]] = []
    unavoidable_cut = False
    start = 0
    while start < length:
        limit = min(start + segment_steps, length)
        end = limit
        if end < length:
            while end > start and end in protected_indices:
                end -= 1
            if end == start:
                # Every possible cut in this interval conflicts for some batch
                # member. Keep the configured memory bound and make progress.
                end = limit
                unavoidable_cut = True
        ranges.append((start, end))
        start = end
    if unavoidable_cut:
        warnings.warn(
            "TBPTT cannot preserve every context-to-supervision transition within "
            "this segment_steps limit and batch alignment; some context writes "
            "receive no gradient from the following supervised frame. Increase "
            "segment_steps or use smaller/aligned batches if that credit is required.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ranges


def iter_segments(batch: Any, segment_steps: int) -> Iterator[Any]:
    """Materialize observations for only the current segment.

    ``segment_steps`` counts sequence observation/memory frames (the S axis),
    including frames with no action prediction target; it does not count action
    chunks or TTT tokens. Demo frame subsampling can change its physical duration.
    ``slice_time`` must retain masks and sequence/session reset information.
    """
    for start, end in segment_ranges(batch, segment_steps):
        yield batch.slice_time(start, end)
