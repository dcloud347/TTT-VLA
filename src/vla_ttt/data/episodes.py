"""Read original/Plus RoboMME HDF5 with explicit observation/action alignment.

Only public RGB, proprioception, task goals and demonstration flags become
model input. Oracle subgoals, grounding and simulator metadata are not exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterator, Sequence

import numpy as np

from vla_ttt.robomme.observations import ACTION_DIM, CONTROL_HZ, make_observation


DATASET_REPO = "Yinpei/robomme_ood_data_h5"
DATASET_REPOS = {"robomme": "Yinpei/robomme_data_h5", "robomme_plus": DATASET_REPO}


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading RoboMME-Plus HDF5 requires h5py; install the project's data dependencies") from exc
    return h5py


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _value(group: Any, key: str, default: Any = None) -> Any:
    return group[key][()] if key in group else default


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    values = np.asarray(value).reshape(-1)
    if not values.size:
        return default
    first = values[0]
    if isinstance(first, (bytes, np.bytes_)):
        first = first.decode("utf-8")
    result = str(first)
    # Some published HDF5 writers store the list of goals as a JSON scalar.
    if result.startswith("["):
        try:
            decoded = json.loads(result)
            if isinstance(decoded, list) and decoded:
                return str(decoded[0])
        except json.JSONDecodeError:
            pass
    return result


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError("Expected a scalar boolean field in episode metadata")
    return bool(scalar[0])


def _read_vector(group: Any, key: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    value = _value(group, key)
    if value is None:
        return np.zeros(size, dtype=np.float32), np.zeros(size, dtype=bool)
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{group.name}/{key}: expected {size} values, got {vector.size}")
    valid = np.isfinite(vector)
    return np.where(valid, vector, 0).astype(np.float32), valid


@dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    group: str
    task_id: str
    instruction: str
    control_hz: float | None
    frame_keys: tuple[str, ...]
    actions: np.ndarray
    action_valid: np.ndarray
    states: np.ndarray
    state_valid: np.ndarray
    is_video_demo: np.ndarray
    source_labels: tuple[str, ...]
    supervised: np.ndarray
    boundaries: np.ndarray
    seed: int | None
    split: str = "train"
    variant: str = "robomme_plus"
    action_offset: int = 0

    @property
    def episode_id(self) -> str:
        return f"{self.task_id}/{self.group}"

    @property
    def length(self) -> int:
        return len(self.frame_keys)


@dataclass(frozen=True)
class ObservationReference:
    episode: EpisodeRecord
    frame: int


def _source_label(step: Any, demo: bool, mode: str) -> str:
    if demo:
        return "video_only"
    explicit = _text(_value(step, "info/action_source")).lower()
    aliases = {"human": "human_correction", "human_action": "human_correction",
               "correction": "human_correction", "policy": "robot", "expert": "demonstration"}
    explicit = aliases.get(explicit, explicit)
    if explicit:
        if explicit not in {"robot", "human_correction", "demonstration"}:
            raise ValueError(f"Unknown action source {explicit!r} in {step.name}")
        return explicit
    for key in ("info/is_human_intervention", "info/is_human_correction"):
        if key in step:
            return "human_correction" if _bool(_value(step, key)) else "robot"
    if mode == "dagger":
        raise ValueError(f"DAgger input needs an action_source/intervention flag at {step.name}")
    return "demonstration"


def read_episode_index(path: Path, group_name: str, *, split: str, mode: str,
                       variant: str = "robomme_plus", observation_timing: str | None = None) -> EpisodeRecord:
    timing = observation_timing or ("post_action" if variant == "robomme" else "pre_action")
    if timing not in {"pre_action", "post_action"}:
        raise ValueError("observation_timing must be pre_action or post_action")
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        episode = handle[group_name]
        declared_timing = _text(_value(episode, "setup/observation_timing"))
        if declared_timing and declared_timing != timing:
            raise ValueError(f"{path}:{group_name}: recorded observation_timing conflicts with configuration")
        frame_keys = tuple(sorted((name for name in episode if name.startswith("timestep_")), key=_natural_key))
        if not frame_keys:
            raise ValueError(f"{path}:{group_name} contains no timestep groups")
        task_id = path.stem.removeprefix("record_dataset_")
        instruction = _text(_value(episode, "setup/task_goal"))
        if not instruction:
            raise ValueError(f"{path}:{group_name} has no setup/task_goal")
        # Original H5 does not promise control_freq. Do not infer it from the
        # replay video's FPS; sequence training operates on recorded frames.
        frequency = _value(episode, "setup/control_freq", CONTROL_HZ if variant == "robomme_plus" else None)
        control_hz = None if frequency is None else float(np.asarray(frequency).item())
        if control_hz is not None and (not np.isfinite(control_hz) or control_hz <= 0):
            raise ValueError(f"Invalid control frequency in {path}:{group_name}")
        raw_seed = _value(episode, "setup/seed")
        seed = None if raw_seed is None else int(np.asarray(raw_seed).item())
        length = len(frame_keys)
        actions = np.zeros((length, ACTION_DIM), dtype=np.float32)
        action_valid = np.zeros_like(actions, dtype=bool)
        states = np.zeros_like(actions)
        state_valid = np.zeros_like(actions, dtype=bool)
        demo = np.zeros(length, dtype=bool)
        supervised = np.zeros_like(actions, dtype=bool)
        boundaries = np.zeros(length, dtype=bool)
        boundaries[0] = True
        sources: list[str] = []
        previous_native: int | None = None
        for index, frame_key in enumerate(frame_keys):
            step = episode[frame_key]
            if "info/is_video_demo" not in step:
                raise ValueError(f"Missing is_video_demo at {step.name}; refusing to guess the phase")
            demo[index] = _bool(_value(step, "info/is_video_demo"))
            joints, joint_valid = _read_vector(step, "obs/joint_state", 7)
            fingers, finger_valid = _read_vector(step, "obs/gripper_state", 2)
            states[index, :7] = joints
            states[index, 7] = fingers.mean()
            state_valid[index, :7] = joint_valid
            state_valid[index, 7] = finger_valid.all()
            actions[index], action_valid[index] = _read_vector(step, "action/joint_action", ACTION_DIM)
            # native_joint_action is intentionally never substituted: MoveCube
            # uses 15 values and stick tasks use 7, with a different contract.
            source = _source_label(step, bool(demo[index]), mode)
            sources.append(source)
            allowed = source == "human_correction" if mode == "dagger" else source in {"human_correction", "demonstration"}
            supervised[index] = action_valid[index] & allowed & ~demo[index]
            if "info/action_loss_mask" in step:
                explicit_mask = np.asarray(_value(step, "info/action_loss_mask"), dtype=bool)
                try:
                    supervised[index] &= np.broadcast_to(explicit_mask, (ACTION_DIM,))
                except ValueError as exc:
                    raise ValueError(f"Invalid action_loss_mask at {step.name}") from exc
            # A phase transition, source transition, missing action, trial reset,
            # or an explicitly recorded control-step jump ends an action chunk.
            native = None
            for key in ("info/native_control_step", "info/native_step", "info/control_step"):
                if key in step:
                    native = int(np.asarray(_value(step, key)).item())
                    break
            if index:
                boundaries[index] = bool(
                    demo[index] != demo[index - 1]
                    or sources[index] != sources[index - 1]
                    or not action_valid[index].all() or not action_valid[index - 1].all()
                    or _bool(_value(step, "info/trial_reset"))
                    or (native is not None and previous_native is not None and native != previous_native + 1)
                )
            previous_native = native
        if np.any(demo & (np.cumsum(~demo) > 0)):
            raise ValueError(f"{path}:{group_name}: demo frames must form a prefix")
        offset = int(timing == "post_action")
        if offset:
            # Original RecordWrapper stores obs AFTER action[t]. Predict the
            # next recorded action, only within a continuous phase/source run.
            next_valid = np.zeros_like(action_valid)
            next_supervised = np.zeros_like(supervised)
            next_actions = np.zeros_like(actions)
            next_valid[:-1] = action_valid[1:] & ~boundaries[1:, None]
            next_supervised[:-1] = supervised[1:] & next_valid[:-1]
            next_actions[:-1] = np.where(next_valid[:-1], actions[1:], 0)
            actions, action_valid, supervised = next_actions, next_valid, next_supervised
            boundaries[1:] |= ~action_valid[1:].all(-1) | ~action_valid[:-1].all(-1)
        return EpisodeRecord(path, group_name, task_id, instruction, control_hz, frame_keys,
                             actions, action_valid, states, state_valid, demo, tuple(sources),
                             supervised, boundaries, seed, split, variant, offset)


def load_observation_grid(rows: Sequence[Sequence[ObservationReference | None]]) -> list[list[dict[str, Any] | None]]:
    """Open each source file once per segment, never keep live h5py handles."""
    from contextlib import ExitStack

    h5py = _h5py()
    output: list[list[dict[str, Any] | None]] = []
    with ExitStack() as stack:
        handles: dict[Path, Any] = {}
        for row in rows:
            observations: list[dict[str, Any] | None] = []
            for reference in row:
                if reference is None:
                    observations.append(None)
                    continue
                record, index = reference.episode, reference.frame
                if record.path not in handles:
                    handles[record.path] = stack.enter_context(h5py.File(record.path, "r"))
                step = handles[record.path][record.group][record.frame_keys[index]]
                images: dict[str, np.ndarray] = {}
                camera_valid: dict[str, bool] = {}
                for camera in ("front", "wrist"):
                    key = f"obs/{camera}_rgb"
                    camera_valid[camera] = key in step
                    # Missing modalities are explicit, not borrowed from a future
                    # frame. The released Plus recordings contain both cameras.
                    images[camera] = np.asarray(step[key][()]) if key in step else np.zeros((256, 256, 3), dtype=np.uint8)
                observation = make_observation(
                    images["front"], images["wrist"], record.states[index], record.instruction,
                    task_id=record.task_id, position=index, is_video_demo=bool(record.is_video_demo[index]),
                    camera_valid=camera_valid, proprio_valid=record.state_valid[index],
                )
                observation["episode_id"] = record.episode_id
                observation["control_hz"] = record.control_hz
                observations.append(observation)
            output.append(observations)
    return output


class RoboMMEDataset:
    """Episode-indexed original/Plus data with deterministic context windows.

    ``context_steps`` counts recorded raw frames, not policy calls or tokens.
    A demonstration prefix is retained in full. If it consumes the whole budget,
    fail explicitly rather than silently returning an unsupervised sequence.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        action_horizon: int = 40,
        observation_stride: int = 40,
        demo_stride: int = 1,
        context_steps: int | None = None,
        split: str = "train",
        split_manifest: str | Path | None = None,
        tasks: Sequence[str] | None = None,
        mode: str = "video",
        seed: int = 0,
        variant: str = "robomme_plus",
        observation_timing: str | None = None,
    ) -> None:
        if variant not in DATASET_REPOS:
            raise ValueError(f"Unknown dataset variant {variant!r}")
        self.variant = variant
        self.dataset_repo = DATASET_REPOS[variant]
        self.observation_timing = observation_timing or ("post_action" if variant == "robomme" else "pre_action")
        if self.observation_timing not in {"pre_action", "post_action"}:
            raise ValueError("observation_timing must be pre_action or post_action")
        if mode not in {"bc", "video", "dagger"}:
            raise ValueError("mode must be 'bc', 'video' or 'dagger'")
        if action_horizon < 1 or observation_stride < 1 or demo_stride < 1:
            raise ValueError("Horizons and strides must be positive")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.observation_stride = int(observation_stride)
        self.demo_stride = int(demo_stride)
        self.split = str(split)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self.set_context_steps(context_steps)
        if self.data_dir.is_file():
            files = [self.data_dir]
        else:
            root = self.data_dir / "h5" if (self.data_dir / "h5").is_dir() else self.data_dir
            files = sorted(root.glob("record_dataset_*.h5"))
        if not files:
            raise FileNotFoundError(f"No record_dataset_<Task>.h5 files under {self.data_dir}; point to the extracted training h5 directory")
        selected_tasks = set(tasks) if tasks else None
        allowed_ids = None
        if split_manifest is not None:
            payload = json.loads(Path(split_manifest).read_text())
            if payload.get("split", split) != split:
                raise ValueError("Requested split disagrees with split_manifest")
            allowed_ids = set(payload["episode_ids"])
        h5py = _h5py()
        episodes = []
        for path in files:
            task = path.stem.removeprefix("record_dataset_")
            if selected_tasks is not None and task not in selected_tasks:
                continue
            with h5py.File(path, "r") as handle:
                names = sorted((name for name in handle if name.startswith("episode_")), key=_natural_key)
            for name in names:
                if allowed_ids is not None and f"{task}/{name}" not in allowed_ids:
                    continue
                record = read_episode_index(path, name, split=split, mode=mode, variant=variant,
                                            observation_timing=self.observation_timing)
                if not record.supervised.any():
                    raise ValueError(f"{record.episode_id} has no supervised execution actions")
                episodes.append(record)
        if not episodes:
            raise ValueError("No episodes match the requested tasks/split manifest")
        self.episodes = tuple(episodes)
        if len({record.episode_id for record in episodes}) != len(episodes):
            raise ValueError("Duplicate task/episode IDs in input HDF5 files")
        if allowed_ids is not None:
            missing = allowed_ids - {record.episode_id for record in episodes}
            if missing:
                raise ValueError(f"split_manifest references missing episodes: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> EpisodeRecord:
        return self.episodes[index]

    def set_context_steps(self, raw_steps: int | None) -> None:
        if raw_steps is not None and raw_steps < 1:
            raise ValueError("context_steps must be positive or None")
        self.context_steps = None if raw_steps is None else int(raw_steps)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "seed": self.seed, "context_steps": self.context_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch, self.seed = int(state["epoch"]), int(state["seed"])
        self.set_context_steps(state.get("context_steps"))

    def window(self, index: int) -> tuple[int, int]:
        record = self.episodes[index]
        demo_length = int(record.is_video_demo.sum())
        first = demo_length if self.mode == "bc" else 0
        if self.context_steps is None or self.context_steps >= record.length - first:
            return first, record.length
        if demo_length and self.mode != "bc":
            if demo_length >= self.context_steps:
                raise ValueError(
                    f"{record.episode_id}: {demo_length} demonstration frames fill context_steps={self.context_steps}; "
                    "increase the raw context budget to retain the complete prefix plus execution"
                )
            return 0, self.context_steps
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        start = int(generator.integers(first, record.length - self.context_steps + 1))
        return start, start + self.context_steps

    def collate(self, indices: Sequence[int]):
        from .collate import collate_episodes
        return collate_episodes(self, indices)

    def iter_batches(self, batch_size: int, **kwargs: Any) -> Iterator[Any]:
        from .sampling import iter_batches
        return iter_batches(self, batch_size, **kwargs)


RoboMMEPlusDataset = RoboMMEDataset  # Backward-compatible public name.
EpisodeDataset = RoboMMEDataset


def validate_dataset(dataset: RoboMMEPlusDataset, *, inspect_images: bool = True) -> dict[str, Any]:
    """Inspect the local data contract; does not run a policy or a simulator."""
    task_counts: dict[str, int] = {}
    supervised_windows = 0
    for index, record in enumerate(dataset.episodes):
        task_counts[record.task_id] = task_counts.get(record.task_id, 0) + 1
        start, end = dataset.window(index)
        supervised_windows += int(record.supervised[start:end].any())
        if inspect_images:
            # Inspect the phase transition as well as the first frame, without
            # reading all images or requiring a model checkpoint.
            candidates = sorted({start, int(record.is_video_demo.sum()), end - 1})
            refs = [[ObservationReference(record, frame) for frame in candidates if start <= frame < end]]
            load_observation_grid(refs)
    return {
        "dataset": dataset.dataset_repo,
        "variant": dataset.variant,
        "observation_timing": dataset.observation_timing,
        "action_offset": int(dataset.observation_timing == "post_action"),
        "split": dataset.split,
        "mode": dataset.mode,
        "episodes": len(dataset),
        "task_episode_counts": task_counts,
        "recorded_frames": sum(record.length for record in dataset.episodes),
        "demo_frames": sum(int(record.is_video_demo.sum()) for record in dataset.episodes),
        "execution_frames": sum(int((~record.is_video_demo).sum()) for record in dataset.episodes),
        "control_hz": sorted({record.control_hz for record in dataset.episodes if record.control_hz is not None}),
        "episodes_without_control_hz": sum(record.control_hz is None for record in dataset.episodes),
        "action_dimension": ACTION_DIM,
        "state_dimension": ACTION_DIM,
        "action_horizon": dataset.action_horizon,
        "decision_stride": dataset.observation_stride,
        "memory_stride": 1,
        "supervised_windows_at_current_epoch": supervised_windows,
        "image_samples_checked": bool(inspect_images),
    }
