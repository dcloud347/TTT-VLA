"""Public RoboMME-Plus observation/action conventions, without oracle fields."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


STATE_FIELDS = ("joint_position", "gripper")
EMBODIMENT_ID = "new_embodiment"
ACTION_DIM = 8
CONTROL_HZ = 20.0
STICK_TASKS = frozenset({"PatternLock", "RouteStick"})


def as_numpy(value: Any, dtype: Any = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def state_vector(joint_state: Any, gripper_state: Any) -> np.ndarray:
    joints = as_numpy(joint_state, np.float32).reshape(-1)
    fingers = as_numpy(gripper_state, np.float32).reshape(-1)
    if joints.size != 7 or fingers.size != 2:
        raise ValueError(f"Expected joint_state[7] and gripper_state[2], got {joints.shape}, {fingers.shape}")
    result = np.r_[joints, np.float32(fingers.mean())].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Robot state contains non-finite values")
    return result


def make_observation(
    front: Any,
    wrist: Any,
    state: Any,
    instruction: str,
    *,
    task_id: str = "",
    position: int = 0,
    is_video_demo: bool = False,
    camera_valid: Mapping[str, bool] | None = None,
    proprio_valid: Any = None,
) -> dict[str, Any]:
    vector = as_numpy(state, np.float32).reshape(-1)
    if vector.size != ACTION_DIM:
        raise ValueError("RoboMME state must have 8 elements")
    images: dict[str, np.ndarray] = {}
    for name, value in (("front", front), ("wrist", wrist)):
        rgb = as_numpy(value)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"{name} must be an RGB uint8 HWC array, got {rgb.shape}, {rgb.dtype}")
        images[name] = np.ascontiguousarray(rgb)
    valid = np.ones(ACTION_DIM, dtype=bool) if proprio_valid is None else np.asarray(proprio_valid, dtype=bool)
    return {
        "images": images,
        "state": {"joint_position": vector[:7].copy(), "gripper": vector[7:].copy()},
        "language": str(instruction),
        "embodiment_id": EMBODIMENT_ID,
        "action_fields": list(STATE_FIELDS),
        "task_id": task_id,
        "position": int(position),
        "is_video_demo": bool(is_video_demo),
        "modality_valid": {
            "images": dict(camera_valid or {"front": True, "wrist": True}),
            "state": {"joint_position": valid[:7].copy(), "gripper": valid[7:].copy()},
        },
    }


def observation_frames(
    obs: Mapping[str, Any], task_goal: str, *, task_id: str = "", start_position: int = 0,
) -> list[dict[str, Any]]:
    """Convert one public env return, retaining every listed observation.

    On reset, all entries except the last are demonstration observations; the
    caller sets that flag because a step return may also contain multiple frames.
    """
    keys = ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list")
    missing = [key for key in keys if key not in obs]
    if missing:
        raise KeyError(f"RoboMME observation is missing {missing}")
    lengths = [len(obs[key]) for key in keys]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"RoboMME observation lists must be nonempty and aligned, got {dict(zip(keys, lengths))}")
    result = []
    for index in range(lengths[0]):
        state = state_vector(obs["joint_state_list"][index], obs["gripper_state_list"][index])
        result.append(make_observation(
            obs["front_rgb_list"][index], obs["wrist_rgb_list"][index], state,
            task_goal, task_id=task_id, position=start_position + index,
        ))
    return result


def standard_joint_action(action: Any) -> np.ndarray:
    """Return the documented public joint_angle action, never native15."""
    values = as_numpy(action, np.float32).reshape(-1)
    if values.size != ACTION_DIM:
        raise ValueError(f"joint_angle requires exactly 8 standard values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Predicted joint action contains non-finite values")
    values = values.copy()
    # The seven joint coordinates stay in physical radians. Only the documented
    # gripper command is bounded; no task-specific solution is inserted here.
    values[-1] = np.clip(values[-1], -1.0, 1.0)
    return values
