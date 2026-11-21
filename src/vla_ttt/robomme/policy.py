"""Bridge the public RoboMME-Plus predict contract to a stateful VLA policy."""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .observations import as_numpy, observation_frames, standard_joint_action


class RoboMMEPlusPolicy:
    """Execute cached chunks while retaining every arriving observation.

    The complete reset demonstration is written first. During chunk execution,
    new observations accumulate on the CPU. Before replanning, all pending past
    frames are written chronologically and the current frame is passed to act
    exactly once. A scene reset between demonstration and execution does not
    reset the learned memory; callers invoke ``reset`` between episodes only.
    """

    def __init__(
        self, policy: Any, *, task_id: str = "", execution_horizon: int | None = None,
    ) -> None:
        self.policy = policy
        self.task_id = task_id
        horizon = int(policy.action_horizon)
        self.execution_horizon = horizon if execution_horizon is None else int(execution_horizon)
        if not 1 <= self.execution_horizon <= horizon:
            raise ValueError("execution_horizon must be in [1, policy.action_horizon]")
        if hasattr(policy, "eval"):
            policy.eval()
        self.reset()

    def reset(self) -> None:
        self.state = None
        self._action_queue: deque[np.ndarray] = deque()
        self._pending_observations: list[dict[str, Any]] = []
        self._first_prediction = True
        self.frames_received = 0
        self.last_error_message: str | None = None

    def predict(
        self, obs: Mapping[str, Any], task_goal: str, error_message: str | None = None,
    ) -> np.ndarray:
        frames = observation_frames(obs, task_goal, task_id=self.task_id, start_position=self.frames_received)
        self.frames_received += len(frames)
        if self.state is None:
            self.state = self.policy.init_state(batch_size=1)
        if self._first_prediction:
            # reset's last frame is already the first execution observation.
            # Do not write it as demo and then again as the current act input.
            for frame in frames[:-1]:
                frame["is_video_demo"] = True
            if len(frames) > 1:
                self.state = self.policy.observe_context(frames[:-1], self.state)
            self._pending_observations.append(frames[-1])
            self._first_prediction = False
        else:
            self._pending_observations.extend(frames)
        if error_message:
            self.last_error_message = str(error_message)
            # A rejected action invalidates the remainder of that open-loop
            # prediction. Keep observations and memory, then replan.
            self._action_queue.clear()
        if not self._action_queue:
            self._plan_chunk()
        return self._action_queue.popleft().copy()

    def _plan_chunk(self) -> None:
        if not self._pending_observations:
            raise RuntimeError("Replanning requires a current observation")
        if len(self._pending_observations) > 1:
            past = self._pending_observations[:-1]
            self.state = self.policy.observe_context(past, self.state)
            # Leave only the current frame if action generation raises, so
            # retrying does not commit the already-consumed context twice.
            self._pending_observations = self._pending_observations[-1:]
        current = self._pending_observations[-1]
        actions, next_state = self.policy.act(current, self.state)
        values = as_numpy(actions, np.float32)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError("RoboMMEPlusPolicy accepts a single online environment")
            values = values[0]
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError(f"Expected physical actions [H,8] or [1,H,8], got {values.shape}")
        if len(values) < self.execution_horizon:
            raise ValueError("The policy produced fewer actions than execution_horizon")
        queue = deque(standard_joint_action(action) for action in values[:self.execution_horizon])
        self.state = next_state
        self._pending_observations.clear()
        self._action_queue = queue

    def flush_context(self) -> None:
        """Consume any final observations already delivered to this wrapper.

        This does not act, step an environment, or start another episode. It is
        optional at episode termination because reset discards that session.
        """
        if self._pending_observations:
            self.state = self.policy.observe_context(self._pending_observations, self.state)
            self._pending_observations.clear()
