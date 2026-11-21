"""Explicit commands: imports never download weights or launch a simulator."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from .config import load_config, validate_config


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _config(args) -> dict:
    config = load_config(args.config)
    if getattr(args, "data", None):
        config.setdefault("data", {})["h5_path"] = args.data
    if getattr(args, "statistics", None):
        config.setdefault("model", {})["statistics_path"] = args.statistics
    if getattr(args, "device", None):
        config.setdefault("model", {})["device"] = args.device
        config.setdefault("training", {})["device"] = args.device
    validate_config(config)
    return config


def _dataset(config: dict, *, full_context: bool = False):
    from .data import RoboMMEDataset

    data = dict(config["data"])
    root = data.pop("h5_path")
    if full_context:
        data["context_steps"] = None
    return RoboMMEDataset(root, **data)


def _build_policy(config: dict, *, inference: bool = False):
    import torch
    from .models.policy import VLATTTPolicy

    runtime = copy.deepcopy(config)
    device = str(runtime["model"].get("device", "cuda"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("The configured GR00T runtime needs CUDA; this machine has no CUDA device")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if "LOCAL_RANK" in os.environ:
            device = f"cuda:{local_rank}"
        torch.cuda.set_device(torch.device(device))
    runtime["model"]["device"] = device
    stats = Path(runtime["model"].get("statistics_path", ""))
    if not stats.is_file():
        raise FileNotFoundError(f"Training statistics are missing: {stats}. Run vla-ttt prepare-data first")
    expected_stats = runtime["model"].get("statistics_dataset")
    if expected_stats:
        from .data.episodes import DATASET_REPOS
        source = json.loads(stats.read_text()).get("dataset")
        if source != DATASET_REPOS[expected_stats]:
            raise ValueError(f"Expected normalization fitted on {expected_stats}, got {source!r}")
    if inference:
        runtime.setdefault("training", {})["stage"] = "inference"
    else:
        os.environ["GR00T_DIT_SDPA_MODE"] = "math"
    torch.manual_seed(int(config.get("training", {}).get("seed", 0)))
    return VLATTTPolicy.from_config(runtime)


def _load_weights(policy, path: str | Path, config: dict) -> dict:
    """Load weights from this project's trusted local training checkpoint."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VLA-TTT checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format_version") != 1 or "model" not in payload:
        raise ValueError("Expected a VLA-TTT checkpoint, not a bare GR00T checkpoint")
    previous = payload.get("config", {})
    if previous.get("ttt", {}) != config.get("ttt", {}):
        raise ValueError("TTT configuration differs from the trained checkpoint (including RoPE/inner LR)")
    for key in ("checkpoint", "revision"):
        if previous.get("model", {}).get(key) != config.get("model", {}).get(key):
            raise ValueError(f"Checkpoint base model identity changed: {key}")
    if bool(previous.get("model", {}).get("use_percentiles", False)) != bool(
        config.get("model", {}).get("use_percentiles", False)
    ):
        raise ValueError("Action normalization mode differs from the trained checkpoint: use_percentiles")
    provenance = payload.get("adapter_provenance", payload.get("provenance", {}))
    current = getattr(policy.adapter, "provenance", {})
    for key in ("github_commit", "cosmos_revision", "statistics_sha256", "reproduction_level"):
        if not provenance.get(key) or provenance.get(key) != current.get(key):
            raise ValueError(f"Checkpoint adapter provenance is missing or differs: {key}")
    policy.load_state_dict(payload["model"], strict=True)
    return payload.get("trainer_state", {})


def command_check_config(args) -> None:
    _print(_config(args))


def command_validate_data(args) -> None:
    from .data import validate_dataset

    _print(validate_dataset(_dataset(_config(args)), inspect_images=not args.skip_images))


def command_prepare_data(args) -> None:
    from .data import validate_dataset, write_statistics

    config = _config(args)
    expected_stats = config["model"].get("statistics_dataset")
    if expected_stats and config["data"].get("variant", "robomme_plus") != expected_stats:
        raise ValueError("Normalization is shared across stages. Run prepare-data with configs/train/pretrain.yaml; use validate-data for Plus")
    dataset = _dataset(config, full_context=True)
    result = validate_dataset(dataset, inspect_images=True)
    destination = args.output or config["model"]["statistics_path"]
    write_statistics(dataset, destination)
    _print({**result, "statistics_path": str(Path(destination).resolve())})


def command_audit(args) -> None:
    config = _config(args)
    if args.load_model:
        policy = _build_policy(config, inference=True)
        result = {**policy.adapter.audit(), "parameters": policy.parameter_summary(), "weights_loaded": True}
    else:
        from .models.adapters.groot import (
            GROOT_COMMIT, GROOT_REVISION, COSMOS_REVISION, UPSTREAM_SOURCE_HASHES,
        )

        result = {"weights_loaded": False, "expected_groot_commit": GROOT_COMMIT,
                  "expected_checkpoint_revision": GROOT_REVISION, "expected_cosmos_revision": COSMOS_REVISION,
                  "source_sha256": UPSTREAM_SOURCE_HASHES, "configuration": config["model"]}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(result)


def command_train(args) -> None:
    from .training import Trainer

    config = _config(args)
    if args.init_checkpoint and config.get("training", {}).get("resume"):
        raise ValueError("--init-checkpoint cannot be combined with training.resume in the configuration")
    if args.output:
        config["training"]["output_dir"] = args.output
    if args.wandb:
        config["training"]["wandb"] = True
    resuming = args.resume or config.get("training", {}).get("resume")
    initialization = args.init_checkpoint or (
        None if resuming else config.get("training", {}).get("init_checkpoint")
    )
    if initialization and not Path(initialization).is_file():
        raise FileNotFoundError(f"Pretraining checkpoint not found: {initialization}. Finish pretrain or supply --init-checkpoint")
    dataset = _dataset(config)
    policy = _build_policy(config)
    if initialization:
        _load_weights(policy, initialization, config)
    from torch import distributed as dist

    existing_group = dist.is_initialized()
    try:
        trainer = Trainer(policy, config)
        result = trainer.fit(dataset, resume=args.resume)
        if trainer.rank == 0:
            _print(result)
    finally:
        if not existing_group and dist.is_initialized():
            dist.destroy_process_group()


def _inference_policy(config: dict, checkpoint: str | None):
    selected = checkpoint or config.get("inference", {}).get("checkpoint")
    if not selected:
        raise ValueError("A trained VLA-TTT --checkpoint is required; raw GR00T weights do not contain TTT")
    if not Path(selected).is_file():
        raise FileNotFoundError(f"VLA-TTT checkpoint not found: {selected}")
    policy = _build_policy(config, inference=True)
    _load_weights(policy, selected, config)
    policy.configure_inference()
    return policy


def command_infer(args) -> None:
    from .robomme import RoboMMEPlusPolicy, run_episode

    config = _config(args)
    settings = config["inference"]
    task = args.task or settings["task"]
    episode = args.episode if args.episode is not None else int(settings.get("episode", 0))
    split = args.split or settings.get("split", "train")
    policy = RoboMMEPlusPolicy(
        _inference_policy(config, args.checkpoint), task_id=task,
        execution_horizon=settings.get("execution_horizon"),
    )
    result = run_episode(policy, task, episode, dataset=split,
                         gui_render=args.gui or bool(settings.get("gui_render", False)))
    # A single requested rollout, not an evaluation suite or leaderboard report.
    _print({"task": task, "episode": episode, "split": split, "status": result.get("status")})


def command_panda(args) -> None:
    from .panda_ros2 import run, validate_settings

    config = _config(args)
    settings = config["panda"]
    validate_settings(settings)
    if args.steps < 1 or args.demo_frames < 0 or not args.instruction.strip():
        raise ValueError("Specify a task instruction, positive step count and nonnegative demo length")
    import rclpy  # Check the sourced ROS environment before loading model weights.
    run(_inference_policy(config, args.checkpoint), settings, args.instruction,
        steps=args.steps, demo_frames=args.demo_frames, execute=args.execute)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-ttt", description="VLA-TTT for RoboMME-Plus")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, default="configs/train/posttrain.yaml"):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", default=default)
        return sub

    check = add("check-config", "Resolve and check YAML without loading weights")
    check.set_defaults(func=command_check_config)
    audit = add("audit-backbone", "Print pinned source metadata or audit a loaded model", "configs/model/groot_public.yaml")
    audit.add_argument("--load-model", action="store_true", help="Explicitly load/download full model weights")
    audit.add_argument("--output")
    audit.add_argument("--statistics")
    audit.add_argument("--device")
    audit.set_defaults(func=command_audit)
    validate = add("validate-data", "Check local RoboMME-Plus HDF5 fields")
    validate.add_argument("--data")
    validate.add_argument("--skip-images", action="store_true")
    validate.set_defaults(func=command_validate_data)
    prepare = add("prepare-data", "Compute shared statistics from original RoboMME training data", "configs/train/pretrain.yaml")
    prepare.add_argument("--data")
    prepare.add_argument("--output")
    prepare.set_defaults(func=command_prepare_data)
    train = add("train", "Run the configured sequence training job")
    train.add_argument("--data")
    train.add_argument("--statistics")
    train.add_argument("--device")
    train.add_argument("--output")
    train.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases on rank 0")
    start = train.add_mutually_exclusive_group()
    start.add_argument("--resume", help="Restore model, optimizer, scheduler, RNG and dataset position")
    start.add_argument("--init-checkpoint", help="Initialize weights only, e.g. pretrain -> posttrain")
    train.set_defaults(func=command_train)
    infer = add("infer", "Run one explicitly selected simulator episode", "configs/inference/default.yaml")
    infer.add_argument("--checkpoint")
    infer.add_argument("--statistics")
    infer.add_argument("--device")
    infer.add_argument("--task")
    infer.add_argument("--episode", type=int)
    infer.add_argument("--split", choices=["train", "test"], default=None)
    infer.add_argument("--gui", action="store_true")
    infer.set_defaults(func=command_infer)
    panda = add("infer-panda", "Predict from a live ROS 2 Panda for one episode", "configs/inference/panda_ros2.yaml")
    panda.add_argument("--checkpoint")
    panda.add_argument("--statistics")
    panda.add_argument("--device")
    panda.add_argument("--instruction", required=True)
    panda.add_argument("--steps", type=int, default=1, help="Number of fresh observations to act on")
    panda.add_argument("--demo-frames", type=int, default=0, help="Live demonstration samples before execution")
    panda.add_argument("--execute", action="store_true", help="Send predicted goals to the Panda controllers")
    panda.set_defaults(func=command_panda)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError, KeyError) as exc:
        print(f"vla-ttt: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
