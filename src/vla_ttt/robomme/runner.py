"""An explicit single-episode integration entry point, without benchmark metrics."""
from __future__ import annotations

from typing import Any, Callable

from .observations import as_numpy
from .policy import RoboMMEPlusPolicy


def _terminal(value: Any) -> bool:
    array = as_numpy(value).reshape(-1)
    if array.size != 1:
        raise ValueError("run_episode only supports one environment")
    return bool(array.item())


def run_episode(
    policy: RoboMMEPlusPolicy,
    task_id: str,
    episode_index: int = 0,
    *,
    dataset: str = "train",
    gui_render: bool = False,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run exactly one requested episode and return the final environment info.

    Importing this module does not import ManiSkill, launch a simulator, or load
    an evaluation split. The default is the public training split. ``env_factory``
    can provide an already-configured environment while retaining this loop.
    """
    if env_factory is None:
        try:
            from robomme_plus import BenchmarkEnvBuilder
        except ImportError as exc:
            raise ImportError(
                "Single-episode rollout requires the official robomme_benchmark installation, including robomme_plus"
            ) from exc
        builder = BenchmarkEnvBuilder(task_id, dataset=dataset, action_space="joint_angle", gui_render=gui_render)
        env = builder.make_env_for_episode(episode_index)
    else:
        env = env_factory()
    try:
        policy.task_id = task_id
        policy.reset()
        obs, info = env.reset()
        goals = info.get("task_goal")
        if isinstance(goals, str):
            task_goal = goals
        elif goals:
            task_goal = str(goals[0])
        else:
            raise ValueError("The environment did not provide task_goal")
        while True:
            action = policy.predict(obs, task_goal, error_message=info.get("error_message"))
            obs, reward, terminated, truncated, info = env.step(action)
            if on_step is not None:
                on_step({"observation": obs, "action": action, "reward": reward,
                         "terminated": terminated, "truncated": truncated, "info": info})
            if _terminal(terminated) or _terminal(truncated):
                return dict(info)
    finally:
        env.close()
