"""Fit normalization only on declared training episodes, without loading RGB."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .episodes import RoboMMEPlusDataset


def _summarize(values: np.ndarray, valid: np.ndarray) -> dict[str, list[float] | list[int]]:
    output: dict[str, list] = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99", "count")}
    for dimension in range(values.shape[-1]):
        column = values[:, dimension][valid[:, dimension]]
        if not len(column):
            raise ValueError(f"No valid training values for normalization dimension {dimension}")
        column = column.astype(np.float64)
        for name, value in (
            ("min", column.min()), ("max", column.max()), ("mean", column.mean()),
            ("std", column.std(ddof=0)), ("q01", np.quantile(column, 0.01)),
            ("q99", np.quantile(column, 0.99)), ("count", len(column)),
        ):
            output[name].append(int(value) if name == "count" else float(value))
    return output


def compute_statistics(dataset: RoboMMEPlusDataset) -> dict[str, Any]:
    if dataset.split != "train" or any(record.split != "train" for record in dataset.episodes):
        raise ValueError("Normalization fitting is restricted to the training split")
    # Each original frame is counted once, independently of action horizon,
    # duplicated sampler rows, context cropping or the current curriculum stage.
    state = np.concatenate([record.states for record in dataset.episodes])
    state_valid = np.concatenate([record.state_valid for record in dataset.episodes])
    action = np.concatenate([record.actions for record in dataset.episodes])
    action_valid = np.concatenate([
        record.supervised & (~record.is_video_demo[:, None]) for record in dataset.episodes
    ])
    return {
        "schema_version": 1,
        "dataset": dataset.dataset_repo,
        "split": "train",
        "episode_ids": [record.episode_id for record in dataset.episodes],
        "state": _summarize(state, state_valid),
        "action": _summarize(action, action_valid),
        "metadata": {
            "format": "robomme-standard-joint-angle-8",
            "variant": dataset.variant,
            "observation_timing": dataset.observation_timing,
            "action_offset": int(dataset.observation_timing == "post_action"),
            "field_order": {"joint_position": [0, 7], "gripper": [7, 8]},
            "state_units": ["radians"] * 7 + ["metres_mean_finger_width"],
            "action_units": ["radians"] * 7 + ["gripper_command_minus1_close_plus1_open"],
            "state_population": "all valid observed training frames, including video prefix",
            "action_population": "supervised execution frames only; each raw frame counted once",
            "training_mode": dataset.mode,
            "recorded_frames": int(state.shape[0]),
            "files": sorted({str(record.path) for record in dataset.episodes}),
            "note": "The caller must supply the official training files or an explicit training split manifest; no test data is discovered or downloaded.",
        },
    }


def write_statistics(dataset: RoboMMEPlusDataset, output_path: str | Path) -> dict[str, Any]:
    statistics = compute_statistics(dataset)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return statistics


# Descriptive alias for CLI integration.
build_normalization_statistics = compute_statistics
