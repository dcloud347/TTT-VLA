"""Optional rank-zero Weights & Biases tracking for a training job."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path

from torch import distributed as dist

from .checkpoint import capture_rng_state, distributed_active, restore_rng_state


@contextmanager
def wandb_run(config: dict, rank: int, world_size: int, output_dir: Path, previous: dict | None):
    training = config.get("training", config)
    if not training.get("wandb", False):
        yield None
        return

    with ExitStack() as stack:
        run = None
        error = None
        if rank == 0:
            rng = capture_rng_state()
            try:
                try:
                    import wandb
                except ImportError as exc:
                    raise RuntimeError("Install the W&B extra with pip install -e '.[wandb]'") from exc

                stage = training.get("stage", "pretrain")
                identity = previous or {}
                run = stack.enter_context(wandb.init(
                    project=identity.get("project") or os.environ.get("WANDB_PROJECT", "vla-ttt"),
                    entity=identity.get("entity") or os.environ.get("WANDB_ENTITY"),
                    id=identity.get("id"),
                    resume="allow" if previous else "never",
                    name=None if previous else os.environ.get("WANDB_NAME", stage),
                    job_type=stage,
                    dir=str(output_dir.resolve()),
                    config={
                        "training": training, "ttt": config.get("ttt", {}),
                        "base_model": config.get("model", {}).get("checkpoint"),
                        "revision": config.get("model", {}).get("revision"),
                        "dataset": config.get("data", {}).get("variant"),
                        "world_size": world_size,
                        "global_batch_size": world_size * int(training.get("batch_size", 1)),
                    },
                ))
            except Exception as exc:
                error = f"Could not start W&B tracking: {exc}"
            finally:
                # Tracking setup may consume random numbers after checkpoint restoration.
                restore_rng_state(rng)
        if distributed_active():
            errors = [error]
            dist.broadcast_object_list(errors, src=0)
            error = errors[0]
        if error:
            raise RuntimeError(error)
        yield run
