"""Small YAML loader with explicit inheritance and environment expansion."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


def merge_dicts(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def replace(match):
            key = match.group(1)
            if key not in os.environ:
                raise ValueError(f"Configuration references unset environment variable {key!r}")
            return os.environ[key]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", replace, value)
    return value


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).expanduser().resolve()
    if path in _seen:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    parents = config.pop("_base_", [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = merge_dicts(merged, load_config(path.parent / parent, (*_seen, path)))
    return _expand(merge_dicts(merged, config))


def validate_config(config: dict) -> None:
    if not isinstance(config.get("training", {}).get("wandb", False), bool):
        raise ValueError("training.wandb must be true or false")
    model = config.get("model", {})
    if not model.get("checkpoint"):
        raise ValueError("model.checkpoint is required")
    if model.get("backbone_alignment", "public_adaptation") != "public_adaptation":
        raise ValueError("Only the pinned public GR00T adapter is implemented; paper-aligned weights are not available")
    ttt = config.get("ttt", {})
    if ttt.get("memory_dim", 1024) % 2:
        raise ValueError("ttt.memory_dim must be even for RoPE")
    for name in ("memory_dim", "fast_hidden_dim", "num_register_tokens"):
        if int(ttt.get(name, {"memory_dim": 1024, "fast_hidden_dim": 1536, "num_register_tokens": 16}[name])) <= 0:
            raise ValueError(f"ttt.{name} must be positive")
    data = config.get("data", {})
    if data.get("variant", "robomme_plus") not in {"robomme", "robomme_plus"}:
        raise ValueError("data.variant must be robomme or robomme_plus")
    if data.get("observation_timing", "pre_action") not in {"pre_action", "post_action"}:
        raise ValueError("data.observation_timing must be pre_action or post_action")
    if model.get("statistics_dataset", "robomme") not in {"robomme", "robomme_plus"}:
        raise ValueError("model.statistics_dataset must be robomme or robomme_plus")
    horizon = int(data.get("action_horizon", 40))
    if horizon != 40 or int(data.get("observation_stride", horizon)) != horizon:
        raise ValueError("The pinned GR00T protocol requires action_horizon=40 and observation_stride=40")
    if float(ttt.get("inner_lr", 0.1)) <= 0 or float(ttt.get("rope_theta", 10000.0)) <= 0:
        raise ValueError("TTT inner_lr and rope_theta must be positive")
    inference = config.get("inference", {})
    execution_horizon = int(inference.get("execution_horizon", horizon))
    if not 1 <= execution_horizon <= 40:
        raise ValueError("inference.execution_horizon must be in [1,40]")
    if execution_horizon != horizon:
        raise ValueError("inference.execution_horizon must match data.action_horizon for this training protocol")
    if int(inference.get("denoising_steps", 4)) <= 0:
        raise ValueError("inference.denoising_steps must be positive")
    if config.get("inference", {}).get("commit_rule", "last_candidate") != "last_candidate":
        raise ValueError("Only inference.commit_rule=last_candidate is supported")

