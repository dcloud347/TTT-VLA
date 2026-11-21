"""GR00T sequence training with TBPTT and optional torchrun data parallelism.

Distributed training deliberately does not wrap the policy in DDP: fast-weight
updates use autograd.grad, while outer gradients are explicitly SUM-reduced once
per logical batch. Every rank uses the same deterministic sampler rounds, even
when its own batch has no supervised actions.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, distributed as dist

from .checkpoint import dataset_signature, distributed_active, load_checkpoint, save_checkpoint
from .flow_matching import (
    FlowMatchingInputs,
    masked_sequence_loss_sum,
    normalized_loss_mask,
    physical_loss_mask,
    sample_flow_matching,
    sequence_denominators,
)
from .tbptt import iter_segments
from .tracking import wandb_run


def make_scheduler(optimizer: torch.optim.Optimizer, settings: Mapping[str, Any], total_steps: int):
    """A configured constant, cosine, or warmup-stable-decay learning-rate schedule."""
    name = str(settings.get("name", "constant")).lower().replace("-", "_")
    if name not in {"constant", "cosine", "wsd", "warmup_stable_decay"}:
        raise ValueError(f"Unknown scheduler: {name}")
    warmup = int(settings.get("warmup_steps", 0))
    decay = int(settings.get("decay_steps", max(total_steps // 10, 1)))
    minimum = float(settings.get("min_lr_ratio", 0.0))
    if warmup < 0 or warmup >= total_steps or not 0.0 <= minimum <= 1.0:
        raise ValueError("Invalid scheduler warmup_steps or min_lr_ratio.")
    if name in {"wsd", "warmup_stable_decay"} and not 0 <= decay <= total_steps - warmup:
        raise ValueError("WSD decay_steps must fit after warmup and before total_steps.")

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if name == "constant" or (name in {"wsd", "warmup_stable_decay"} and decay == 0):
            return 1.0
        start = warmup if name == "cosine" else total_steps - decay
        if step < start:
            return 1.0
        progress = min(max((step - start) / max(total_steps - start, 1), 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def epoch_batch_indices(
    length: int,
    *,
    batch_size: int,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    drop_last: bool,
) -> list[list[int]]:
    """Deterministic, equal-length per-rank rounds, padded with repeated indices.

    Padding is explicit: the final round may repeat examples, as with a standard
    DistributedSampler. Set drop_last to avoid repeated examples in each epoch.
    """
    if length <= 0 or batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid dataset length, batch size, or distributed rank.")
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(length, generator=generator).tolist() if shuffle else list(range(length))
    round_size = batch_size * world_size
    rounds = length // round_size if drop_last else math.ceil(length / round_size)
    if rounds == 0:
        raise ValueError("drop_last would discard the entire dataset; reduce batch_size or disable it.")
    required = rounds * round_size
    if required > length:
        indices = (indices * math.ceil(required / length))[:required]
    else:
        indices = indices[:required]
    return [
        indices[start + rank * batch_size : start + (rank + 1) * batch_size]
        for start in range(0, required, round_size)
    ]


class Trainer:
    """Train a policy using lazy physical-action trajectory batches.

    Expected policy interface: configure_trainable(stage), init_state(batch_size,
    device), normalize_actions(actions, observations), forward_sequence(segment,
    state, noisy_actions, tau), and detach_state(state, requires_grad=True).
    The sequence output exposes ``velocity`` and ``state``.
    """

    def __init__(self, policy: torch.nn.Module, config: Mapping[str, Any]):
        # Fused SDPA backwards may not support the derivatives required by the
        # fast-weight inner loop. The CLI sets this before loading GR00T too.
        os.environ["GR00T_DIT_SDPA_MODE"] = "math"
        self.config = dict(config)
        self.settings = dict(config.get("training", config))
        self.policy = policy
        self.total_steps = int(self.settings.get("total_steps", 0))
        self.batch_size = int(self.settings.get("batch_size", 1))
        self.segment_steps = int(self.settings.get("segment_steps", 0))
        if self.total_steps <= 0 or self.batch_size <= 0 or self.segment_steps <= 0:
            raise ValueError("training.total_steps, batch_size, and segment_steps must be positive.")
        self.seed = int(self.settings.get("seed", 0))
        self.precision = str(self.settings.get("precision", "fp32")).lower()
        if self.precision not in {"fp32", "float32", "bf16", "bfloat16"}:
            raise ValueError("Training precision must be fp32 or bf16; FP16 scaling is not enabled.")
        self.device = self._initialize_device()
        self.rank = dist.get_rank() if distributed_active() else 0
        self.world_size = dist.get_world_size() if distributed_active() else 1
        self.policy.to(self.device)
        stage = str(self.settings.get("stage", "pretrain"))
        if stage not in {"pretrain", "posttrain"}:
            raise ValueError("training.stage must be pretrain or posttrain.")
        self.policy.configure_trainable(stage)
        self.parameters = [p for p in self.policy.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable parameters after configure_trainable().")
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=float(self.settings.get("learning_rate", 2e-5 if stage == "pretrain" else 5e-5)),
            weight_decay=float(self.settings.get("weight_decay", 1e-5)),
            betas=tuple(self.settings.get("betas", (0.9, 0.999))),
        )
        scheduler_settings = self.settings.get("scheduler", {"name": "constant"})
        if isinstance(scheduler_settings, str):
            scheduler_settings = {"name": scheduler_settings}
        self.scheduler = make_scheduler(self.optimizer, scheduler_settings, self.total_steps)
        self.output_dir = Path(self.settings.get("output_dir", "outputs/train"))
        self.global_step = 0
        self.epoch = 0
        self.batch_index = 0
        self._active_context_steps: int | None = None
        self.context_schedule = self._parse_context_schedule()
        self._wandb_run = None
        self._wandb_identity: dict | None = None

    def _initialize_device(self) -> torch.device:
        requested = str(self.settings.get("device", "auto"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        if device.type == "cuda":
            index = int(os.environ.get("LOCAL_RANK", device.index if device.index is not None else 0))
            torch.cuda.set_device(index)
            device = torch.device("cuda", index)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not distributed_active():
            timeout = int(self.settings.get("distributed_timeout_seconds", 3600))
            if timeout <= 0:
                raise ValueError("distributed_timeout_seconds must be positive")
            # Ranks process whole episodes of different lengths before reducing.
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=timedelta(seconds=timeout),
            )
        return device

    def _parse_context_schedule(self) -> list[tuple[int, int]]:
        schedule = []
        for item in self.settings.get("context_schedule", []):
            step, length = int(item["step"]), int(item["raw_context_steps"])
            if step < 0 or length <= 0:
                raise ValueError("Context schedule needs nonnegative steps and positive raw_context_steps.")
            schedule.append((step, length))
        if schedule and (schedule[0][0] != 0 or any(a[0] >= b[0] for a, b in zip(schedule, schedule[1:]))):
            raise ValueError("context_schedule must start at step 0 and have strictly increasing steps.")
        return schedule

    def _apply_context_schedule(self, dataset: Any) -> None:
        length = None
        for step, candidate in self.context_schedule:
            if step <= self.global_step:
                length = candidate
        if length is not None and length != self._active_context_steps:
            if not hasattr(dataset, "set_context_steps"):
                raise TypeError("Context curriculum requires dataset.set_context_steps(raw_steps).")
            dataset.set_context_steps(length)
            self._active_context_steps = length

    def _seed_rngs(self) -> None:
        seed = self.seed + self.rank
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:
            pass

    def _synchronize_initial_weights(self) -> None:
        if distributed_active():
            with torch.no_grad():
                for parameter in self.policy.parameters():
                    dist.broadcast(parameter, src=0)
                for buffer in self.policy.buffers():
                    dist.broadcast(buffer, src=0)

    def _sum_outer_gradients(self) -> None:
        """Preserve globally-unused grad=None, and participate on unsupervised ranks."""
        if not distributed_active():
            return
        present = torch.tensor([p.grad is not None for p in self.parameters], dtype=torch.int32, device=self.device)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        for parameter, has_grad in zip(self.parameters, present.tolist()):
            if not has_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if parameter.grad.is_sparse:
                raise TypeError("Manual outer-gradient reduction requires dense gradients.")
            # Loss already uses the global supervised-sequence denominator.
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    def train_batch(self, batch: Any) -> dict[str, Any]:
        """One logical-batch update, normalizing actions only for its current segment."""
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        denominators = sequence_denominators(batch).to(self.device)
        local_sequences = (denominators > 0).sum()
        global_sequences = local_sequences.clone()
        if distributed_active():
            dist.all_reduce(global_sequences, op=dist.ReduceOp.SUM)
        count = int(global_sequences.item())
        if count == 0:
            return {"updated": False, "loss": None, "supervised_sequences": 0}

        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        if int(local_sequences.item()) > 0:
            state = self.policy.init_state(batch.batch_size, device=self.device)
            for segment in iter_segments(batch, self.segment_steps):
                segment = segment.to(self.device)
                with torch.no_grad():
                    normalized_actions, normalization_valid = self.policy.normalize_actions(
                        segment.actions, segment.observations
                    )
                    normalized_actions = normalized_actions.to(self.device)
                    mask = normalized_loss_mask(
                        physical_loss_mask(segment), normalized_actions, normalization_valid
                    )
                    flow = sample_flow_matching(normalized_actions)
                    # Video prefixes and intermediate observed-only frames use
                    # exactly the same placeholder convention as observe_context.
                    has_actions = segment.action_valid_mask.bool().any(dim=(-1, -2))
                    has_actions = has_actions & segment.valid_mask.bool()
                    flow = FlowMatchingInputs(
                        noisy_actions=torch.where(has_actions[..., None, None], flow.noisy_actions, 0.0),
                        tau=torch.where(has_actions, flow.tau, 0.0),
                        target_velocity=flow.target_velocity,
                    )
                autocast = (
                    torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                    if self.precision in {"bf16", "bfloat16"} else nullcontext()
                )
                with autocast:
                    output = self.policy.forward_sequence(
                        segment, state, noisy_actions=flow.noisy_actions, tau=flow.tau
                    )
                    segment_loss = masked_sequence_loss_sum(
                        output.velocity, flow.target_velocity, mask, denominators
                    )
                if bool(mask.any().item()):
                    (segment_loss / count).backward()
                loss_sum.add_(segment_loss.detach())
                state = self.policy.detach_state(output.state, requires_grad=True)
                # Do not retain the previous segment's decoded images or graph
                # while the next slice is materialized.
                del output, segment_loss, flow, normalized_actions, normalization_valid, mask, segment
            del state

        self._sum_outer_gradients()
        active_parameters = [p for p in self.parameters if p.grad is not None]
        if not active_parameters:
            raise RuntimeError("Supervised actions produced no outer parameter gradients.")
        max_norm = self.settings.get("max_grad_norm")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active_parameters,
            float(max_norm) if max_norm is not None else float("inf"),
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        if distributed_active():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        return {
            "updated": True,
            "loss": float((loss_sum / count).item()),
            "supervised_sequences": count,
            "grad_norm": float(gradient_norm.item()),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _state(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            **({"wandb_run": self._wandb_identity} if self._wandb_identity else {}),
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "active_context_steps": self._active_context_steps,
        }

    def _save(self, dataset: Any) -> Path:
        return save_checkpoint(
            self.output_dir / "checkpoints" / "latest.pt",
            policy=self.policy,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            trainer_state=self._state(),
            dataset=dataset,
        )

    def _log(self, values: Mapping[str, Any]) -> None:
        if self.rank != 0:
            return
        record = {"step": self.global_step, "epoch": self.epoch, "batch_index": self.batch_index, **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (self.output_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        # A resumed online run may already contain metrics newer than our checkpoint.
        run = self._wandb_run
        if run is not None and values.get("updated") and self.global_step >= run.starting_step:
            run.log(record, step=self.global_step, commit=True)

    def fit(self, dataset: Any, resume: str | Path | None = None) -> dict[str, Any]:
        """Run configured training; no evaluation, synthetic tasks, or experiment sweep."""
        if len(dataset) == 0:
            raise ValueError("Cannot train on an empty dataset.")
        if distributed_active():
            signatures: list[Any] = [None] * self.world_size
            dist.all_gather_object(signatures, dataset_signature(dataset))
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise ValueError("All ranks must see the same ordered dataset before index sharding.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seed_rngs()
        resume_path = resume or self.settings.get("resume")
        self._wandb_identity = None
        if resume_path:
            if resume_path is True:
                resume_path = self.output_dir / "checkpoints" / "latest.pt"
            metadata = load_checkpoint(
                resume_path,
                policy=self.policy,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                dataset=dataset,
                strict=bool(self.settings.get("resume_strict", True)),
                expected_config=self.config,
            )
            state = metadata["trainer_state"]
            self._wandb_identity = state.get("wandb_run")
            self.global_step = int(state["global_step"])
            self.epoch = int(state["epoch"])
            self.batch_index = int(state["batch_index"])
            if min(self.global_step, self.epoch, self.batch_index) < 0:
                raise ValueError("Checkpoint contains a negative training or sampler position.")
            self._active_context_steps = None  # Reapply curriculum to the restored dataset.
        self._synchronize_initial_weights()
        with wandb_run(self.config, self.rank, self.world_size, self.output_dir, self._wandb_identity) as run:
            if run is not None:
                self._wandb_identity = {"id": run.id, "project": run.project, "entity": run.entity}
            self._wandb_run = run
            try:
                return self._fit_epochs(dataset)
            finally:
                self._wandb_run = None

    def _fit_epochs(self, dataset: Any) -> dict[str, Any]:
        self._log({
            "event": "training_started",
            "world_size": self.world_size,
            "per_rank_batch_size": self.batch_size,
            "global_batch_size": self.batch_size * self.world_size,
            "distributed_backend": str(dist.get_backend()) if distributed_active() else None,
            "stage": self.settings.get("stage", "pretrain"),
        })
        epochs = self.settings.get("epochs")
        max_epochs = int(epochs) if epochs is not None else None
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("epochs must be positive when specified.")
        checkpoint_every = int(self.settings.get("checkpoint_every", 1000))
        log_every = int(self.settings.get("log_every", 10))
        if checkpoint_every < 0 or log_every <= 0:
            raise ValueError("checkpoint_every must be nonnegative and log_every must be positive.")
        while self.global_step < self.total_steps and (max_epochs is None or self.epoch < max_epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.epoch)
            indices = epoch_batch_indices(
                len(dataset), batch_size=self.batch_size, world_size=self.world_size, rank=self.rank,
                seed=self.seed, epoch=self.epoch, shuffle=bool(self.settings.get("shuffle", True)),
                drop_last=bool(self.settings.get("drop_last", False)),
            )
            if self.batch_index > len(indices):
                raise ValueError("Checkpoint batch_index exceeds this epoch's sampler length.")
            # A checkpoint after the last batch of an epoch resumes at the next epoch.
            if self.batch_index == len(indices):
                self.epoch += 1
                self.batch_index = 0
                continue
            started_at_epoch_boundary = self.batch_index == 0
            updates_this_epoch = 0
            while self.batch_index < len(indices) and self.global_step < self.total_steps:
                self._apply_context_schedule(dataset)
                started = time.monotonic()
                batch = dataset.collate(indices[self.batch_index])
                result = self.train_batch(batch)
                self.batch_index += 1
                if result["updated"]:
                    updates_this_epoch += 1
                    if self.global_step % log_every == 0:
                        self._log({**result, "seconds": time.monotonic() - started})
                    if checkpoint_every and self.global_step % checkpoint_every == 0:
                        self._save(dataset)
                else:
                    self._log(result)
                del batch
            if self.batch_index == len(indices):
                if updates_this_epoch == 0 and started_at_epoch_boundary:
                    raise RuntimeError("No supervised optimizer updates in a full epoch; inspect action masks.")
                self.epoch += 1
                self.batch_index = 0
        checkpoint_path = self._save(dataset)
        return {**self._state(), "checkpoint": str(checkpoint_path), "total_steps_reached": self.global_step >= self.total_steps}
