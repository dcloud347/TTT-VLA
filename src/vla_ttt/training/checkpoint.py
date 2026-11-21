"""Atomic, logical-batch-boundary checkpoints, including each rank's RNG state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import distributed as dist


FORMAT_VERSION = 1


def distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def dataset_signature(dataset: Any) -> dict[str, Any] | None:
    """Identify sampler order and raw episode lengths without reading RGB frames.

    This is an ordering identity, not a checksum of every video/data byte. Paths
    are deliberately excluded so an identical dataset can be relocated.
    """
    if dataset is None:
        return None
    signature: dict[str, Any] = {"length": len(dataset)}
    if hasattr(dataset, "variant"):
        signature["variant"] = dataset.variant
    episodes = getattr(dataset, "episodes", None)
    if episodes is not None:
        digest = hashlib.sha256()
        for record in episodes:
            entry = [str(record.episode_id), int(record.length)]
            digest.update(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        signature["ordered_episodes_sha256"] = digest.hexdigest()
    return signature


def _adapter_provenance(policy: torch.nn.Module) -> dict[str, Any]:
    return dict(getattr(getattr(policy, "adapter", None), "provenance", None) or {})


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu(), device=torch.cuda.current_device())


def save_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
    dataset: Any = None,
) -> Path:
    """All ranks must call this together; only rank zero writes the file.

    Saving between logical batches avoids serializing live higher-order graphs
    or partially accumulated gradients. Runtime fast weights are per-sequence
    state and are reinitialized for the next logical batch.
    """
    target = Path(path)
    distributed = distributed_active()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_state = {
        "rng": capture_rng_state(),
        "dataset": dataset.state_dict() if hasattr(dataset, "state_dict") else None,
    }
    rank_states: list[Any] = [None] * world_size
    if distributed:
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states[0] = local_state

    error: str | None = None
    if rank == 0:
        temporary: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "format_version": FORMAT_VERSION,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": dict(config),
                "adapter_provenance": _adapter_provenance(policy),
                "dataset_signature": dataset_signature(dataset),
                "trainer_state": dict(trainer_state),
                "world_size": world_size,
                "rank_states": rank_states,
            }
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            os.close(fd)
            torch.save(payload, temporary)
            os.replace(temporary, target)
            temporary = None
        except Exception as exc:
            error = f"Could not save checkpoint {target}: {type(exc).__name__}: {exc}"
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    if distributed:
        messages = [error]
        dist.broadcast_object_list(messages, src=0)
        error = messages[0]
    if error is not None:
        raise RuntimeError(error)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    dataset: Any = None,
    strict: bool = True,
    restore_random_state: bool = True,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a trusted local training checkpoint, including optimizer state.

    These checkpoints contain Python/NumPy RNG objects, so this deliberately uses
    pickle-capable loading. Do not use it with untrusted checkpoint files.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported VLA-TTT checkpoint format.")
    world_size = dist.get_world_size() if distributed_active() else 1
    rank = dist.get_rank() if distributed_active() else 0
    if strict and payload["world_size"] != world_size:
        raise ValueError("Exact resume requires the original world size.")
    if strict and expected_config is not None:
        previous = payload["config"].get("training", payload["config"])
        current = expected_config.get("training", expected_config)
        for key in (
            "stage", "batch_size", "segment_steps", "seed", "shuffle", "drop_last",
            "context_schedule", "scheduler", "total_steps", "precision",
            "learning_rate", "weight_decay", "betas", "max_grad_norm", "epochs",
        ):
            if previous.get(key) != current.get(key):
                raise ValueError(f"Resume configuration changed training.{key}; use an explicit non-strict restart.")
        for key in ("model", "data", "ttt"):
            if payload["config"].get(key, {}) != expected_config.get(key, {}):
                raise ValueError(f"Resume configuration changed {key}.")
        # These affect state evolution or prediction but need not change tensor
        # shapes, so strict state_dict loading alone cannot detect a mismatch.
        saved_inference = payload["config"].get("inference", {})
        current_inference = expected_config.get("inference", {})
        for key, default in (("commit_rule", "last_candidate"), ("denoising_steps", 4), ("execution_horizon", 40)):
            if saved_inference.get(key, default) != current_inference.get(key, default):
                raise ValueError(f"Resume configuration changed inference.{key}.")
    if strict:
        saved_provenance = payload.get("adapter_provenance", {})
        current_provenance = _adapter_provenance(policy)
        for key in (
            "checkpoint", "revision", "github_commit", "cosmos_revision", "statistics_sha256",
            "reproduction_level",
        ):
            if saved_provenance.get(key) != current_provenance.get(key):
                raise ValueError(f"Resume adapter provenance changed {key}.")
        if dataset is not None and payload.get("dataset_signature") != dataset_signature(dataset):
            raise ValueError("Resume dataset episode count, order, or lengths changed.")
    policy.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    rank_state = payload["rank_states"][rank % len(payload["rank_states"])]
    if dataset is not None and rank_state["dataset"] is not None:
        if not hasattr(dataset, "load_state_dict"):
            raise ValueError("Checkpoint has dataset state, but dataset cannot restore it.")
        dataset.load_state_dict(rank_state["dataset"])
    if restore_random_state:
        restore_rng_state(rank_state["rng"])
    return {
        "trainer_state": payload["trainer_state"],
        "config": payload["config"],
        "world_size": payload["world_size"],
        "adapter_provenance": payload.get("adapter_provenance", {}),
        "dataset_signature": payload.get("dataset_signature"),
    }
