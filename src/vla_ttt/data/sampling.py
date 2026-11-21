"""Deterministic, equal-length distributed episode batches."""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from .episodes import RoboMMEPlusDataset
from .schema import TrajectoryBatch


def batch_indices(
    length: int, batch_size: int, *, shuffle: bool = True, seed: int = 0,
    epoch: int = 0, rank: int = 0, world_size: int = 1, drop_last: bool = False,
) -> Iterator[list[int]]:
    """Pad the global order before sharding so every rank has equal work.

    The duplicated tail is explicit deterministic sampler padding, not temporal
    padding inside an episode. ``drop_last=True`` avoids repeated episodes.
    """
    if length < 1 or batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid dataset length, batch size or distributed rank")
    order = np.arange(length, dtype=np.int64)
    if shuffle:
        np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)])).shuffle(order)
    global_size = batch_size * world_size
    if drop_last:
        total = (length // global_size) * global_size
        order = order[:total]
    else:
        total = math.ceil(length / global_size) * global_size
        order = np.resize(order, total)
    for start in range(0, total, global_size):
        left = start + rank * batch_size
        yield order[left:left + batch_size].tolist()


def iter_batches(
    dataset: RoboMMEPlusDataset, batch_size: int, *, shuffle: bool = True,
    seed: int | None = None, epoch: int | None = None, rank: int = 0,
    world_size: int = 1, drop_last: bool = False, start_batch: int = 0,
) -> Iterator[TrajectoryBatch]:
    if start_batch < 0:
        raise ValueError("start_batch must be nonnegative")
    selected_epoch = dataset.epoch if epoch is None else int(epoch)
    dataset.set_epoch(selected_epoch)
    for batch_index, indices in enumerate(batch_indices(
        len(dataset), batch_size, shuffle=shuffle,
        seed=dataset.seed if seed is None else seed, epoch=selected_epoch,
        rank=rank, world_size=world_size, drop_last=drop_last,
    )):
        if batch_index >= start_batch:
            yield dataset.collate(indices)
