"""Build action chunks while keeping intermediate observations as context."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .episodes import ObservationReference, RoboMMEPlusDataset, load_observation_grid
from .schema import LazyObservationGrid, TrajectoryBatch


def collate_episodes(dataset: RoboMMEPlusDataset, indices: Sequence[int]) -> TrajectoryBatch:
    if not indices:
        raise ValueError("Cannot collate an empty batch")
    plans = []
    for dataset_index in indices:
        record = dataset.episodes[int(dataset_index)]
        start, end = dataset.window(int(dataset_index))
        frames = [index for index in range(start, end)
                  if not record.is_video_demo[index] or index % dataset.demo_stride == 0]
        if not frames:
            raise ValueError(f"Empty context window for {record.episode_id}")
        # Restart the decision cadence at every phase/source/action boundary.
        # All other execution frames remain in the sequence as memory-only input.
        boundaries = [start] + [index for index in range(start + 1, end) if record.boundaries[index]] + [end]
        decisions: dict[int, int] = {}
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if not record.is_video_demo[left]:
                decisions.update({index: right for index in range(left, right, dataset.observation_stride)})
        plans.append((record, start, end, frames, decisions))

    batch_size, steps = len(plans), max(len(plan[3]) for plan in plans)
    horizon = dataset.action_horizon
    shape = (batch_size, steps, horizon, 8)
    actions = np.zeros(shape, dtype=np.float32)
    action_valid = np.zeros(shape, dtype=bool)
    action_loss = np.zeros(shape, dtype=bool)
    valid = np.zeros((batch_size, steps), dtype=bool)
    reset = np.zeros_like(valid)
    demo = np.zeros_like(valid)
    positions = np.zeros((batch_size, steps), dtype=np.int64)
    rows = []
    labels = []
    metadata = []
    ids = []
    for batch_index, (record, start, end, frames, decisions) in enumerate(plans):
        references: list[ObservationReference | None] = []
        sources = []
        for position, frame in enumerate(frames):
            valid[batch_index, position] = True
            demo[batch_index, position] = record.is_video_demo[frame]
            positions[batch_index, position] = frame
            references.append(ObservationReference(record, frame))
            sources.append(record.source_labels[frame])
            if frame in decisions:
                stop = min(frame + horizon, decisions[frame], end)
                count = stop - frame
                actions[batch_index, position, :count] = record.actions[frame:stop]
                action_valid[batch_index, position, :count] = record.action_valid[frame:stop]
                action_loss[batch_index, position, :count] = record.supervised[frame:stop]
        reset[batch_index, 0] = True
        references.extend([None] * (steps - len(references)))
        sources.extend(["padding"] * (steps - len(sources)))
        rows.append(references)
        labels.append(tuple(sources))
        ids.append(record.episode_id)
        metadata.append({
            "task_id": record.task_id,
            "variant": record.variant,
            "action_offset": record.action_offset,
            "episode_id": record.episode_id,
            "raw_window": [start, end],
            "raw_context_steps": end - start,
            "recorded_episode_steps": record.length,
            "control_hz": record.control_hz,
            "action_horizon": horizon,
            "decision_stride": dataset.observation_stride,
            "memory_stride": 1,
            "demo_stride": dataset.demo_stride,
            "source_file": str(record.path),
            "split": record.split,
        })
    action_loss &= action_valid
    normalizer = action_loss.sum(axis=(1, 2, 3)).astype(np.float32)
    # A DAgger crop can contain only autonomous actions on one distributed rank.
    # Return it so every rank still reaches the same synchronization points.
    # The trainer decides whether the GLOBAL batch has any supervision.
    return TrajectoryBatch(
        actions=torch.from_numpy(actions),
        action_loss_mask=torch.from_numpy(action_loss),
        action_valid_mask=torch.from_numpy(action_valid),
        valid_mask=torch.from_numpy(valid),
        state_update_mask=torch.from_numpy(valid.copy()),
        reset_mask=torch.from_numpy(reset),
        positions=torch.from_numpy(positions),
        observations=LazyObservationGrid(rows, load_observation_grid),
        loss_normalizer=torch.from_numpy(normalizer),
        sequence_supervised_mask=torch.from_numpy(normalizer > 0),
        episode_ids=tuple(ids),
        is_video_demo=torch.from_numpy(demo),
        source_labels=tuple(labels),
        metadata=tuple(metadata),
    )
