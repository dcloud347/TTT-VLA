"""Single-episode Panda rollout through ROS 2 JointTrajectoryController and Franka Hand."""
from __future__ import annotations

import json
import time

import numpy as np

from .robomme.observations import as_numpy, make_observation, standard_joint_action, state_vector


# Franka Emika Panda joint limits in manufacturer joint order (rad).
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def validate_settings(cfg):
    for key in ("max_sensor_skew", "max_sensor_age", "max_prediction_age", "max_joint_step",
                "trajectory_duration", "gripper_max_effort"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"panda.{key} must be finite and positive")
    if not 0 <= cfg["close_finger_position"] < cfg["open_finger_position"] <= 0.04:
        raise ValueError("Expected ordered Panda finger positions in [0, 0.04] meters")
    arms, fingers = cfg["arm_joints"], cfg["finger_joints"]
    if len(arms) != 7 or len(fingers) != 2 or len(set(arms + fingers)) != 9:
        raise ValueError("Specify seven ordered arm and two distinct finger joint names")
    if cfg["front_image"] == cfg["wrist_image"]:
        raise ValueError("Front and wrist images must come from different cameras")


def rgb_image(msg):
    if msg.encoding not in ("rgb8", "bgr8") or min(msg.height, msg.width) <= 0:
        raise ValueError(f"Expected a nonempty rgb8 or bgr8 sensor_msgs/Image, got {msg.encoding}")
    if msg.step < 3 * msg.width or len(msg.data) != msg.height * msg.step:
        raise ValueError("Camera payload length or row stride is invalid")
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :3 * msg.width].reshape(msg.height, msg.width, 3)
    return np.ascontiguousarray(image if msg.encoding == "rgb8" else image[:, :, ::-1])


class Sensors:
    def __init__(self, node, cfg):
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState

        self.node = node
        self.cfg = cfg
        self.latest = {}
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for key in ("front_image", "wrist_image"):
            node.create_subscription(Image, cfg[key], lambda msg, key=key: self.save(key, msg, msg), qos)
        # Works with a merged joint_states topic or separate arm/hand publishers.
        for topic in {cfg["joint_states"], cfg["finger_states"]}:
            node.create_subscription(JointState, topic, lambda msg, topic=topic: self.joints(topic, msg), qos)

    def save(self, key, msg, value):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if stamp > self.latest.get(key, (0,))[0]:
            self.latest[key] = stamp, value

    def joints(self, topic, msg):
        if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
            raise ValueError("Malformed JointState names/positions")
        positions = dict(zip(msg.name, msg.position))
        for key, source in (("arm_joints", "joint_states"), ("finger_joints", "finger_states")):
            names = self.cfg[key]
            if self.cfg[source] == topic and all(name in positions for name in names):
                self.save(key, msg, [positions[name] for name in names])

    def sample(self):
        import rclpy

        clock = self.node.get_clock()
        start = clock.now().nanoseconds
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if len(self.latest) != 4:
                continue
            stamps = [entry[0] for entry in self.latest.values()]
            now = clock.now().nanoseconds
            if min(stamps) <= start or max(stamps) > now:
                continue
            if (now - min(stamps)) / 1e9 > self.cfg["max_sensor_age"]:
                continue
            if (max(stamps) - min(stamps)) / 1e9 <= self.cfg["max_sensor_skew"]:
                values = {key: pair[1] for key, pair in self.latest.items()}
                state = state_vector(values["arm_joints"], values["finger_joints"])
                return min(stamps), rgb_image(values["front_image"]), rgb_image(values["wrist_image"]), state
        raise RuntimeError("Timed out waiting for fresh aligned Panda observations. Check topics, joint names and clocks")


def check_target(action, current_q, max_step):
    if np.any(action[:7] < Q_MIN) or np.any(action[:7] > Q_MAX):
        raise ValueError("Prediction exceeds Panda joint limits")
    if np.max(np.abs(action[:7] - current_q)) > max_step:
        raise ValueError("Prediction exceeds max_joint_step from measured pose")


class Controllers:
    def __init__(self, node, cfg):
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from rclpy.action import ActionClient

        self.node = node
        self.cfg = cfg
        self.arm = ActionClient(node, FollowJointTrajectory, cfg["arm_action"])
        self.hand = ActionClient(node, GripperCommand, cfg["gripper_action"])
        self.last_open = None
        for client in (self.arm, self.hand):
            if not client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError("Panda action server unavailable. Check active controller names and action types")

    def wait(self, future, timeout):
        import rclpy

        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError("Timed out waiting for Panda action server")
        return future.result()

    def send(self, client, goal):
        from action_msgs.msg import GoalStatus

        pending = client.send_goal_async(goal)
        handle = None
        try:
            handle = self.wait(pending, 5.0)
            if not handle.accepted:
                raise RuntimeError("Controller rejected Panda goal")
            result = self.wait(handle.get_result_async(), max(10.0, self.cfg["trajectory_duration"] + 5))
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"Panda goal ended with status {result.status}: {result.result}")
            return result.result
        except (KeyboardInterrupt, RuntimeError):
            if handle is None:
                # An interrupted acknowledgement can still yield an accepted goal.
                handle = self.wait(pending, 2.0)
            if handle.accepted:
                self.wait(handle.cancel_goal_async(), 2.0)
            raise

    def execute(self, action):
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from rclpy.duration import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.cfg["arm_joints"]
        point = JointTrajectoryPoint()
        point.positions = action[:7].tolist()
        point.velocities = [0.0] * 7
        point.time_from_start = Duration(seconds=self.cfg["trajectory_duration"]).to_msg()
        goal.trajectory.points = [point]
        result = self.send(self.arm, goal)
        if result.error_code != result.SUCCESSFUL:
            raise RuntimeError(f"Panda trajectory failed: {result.error_string}")

        opened = bool(action[7] >= 0)
        if opened != self.last_open:
            grip = GripperCommand.Goal()
            key = "open_finger_position" if opened else "close_finger_position"
            grip.command.position = float(self.cfg[key])
            grip.command.max_effort = float(self.cfg["gripper_max_effort"])
            result = self.send(self.hand, grip)
            if not result.reached_goal:
                raise RuntimeError("Franka Hand did not complete gripper command")
            self.last_open = opened


def run(policy, cfg, instruction, *, steps, demo_frames, execute):
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("vla_ttt_panda")
    try:
        sensors = Sensors(node, cfg)
        controllers = Controllers(node, cfg) if execute else None
        memory = policy.init_state(batch_size=1)
        if demo_frames:
            node.get_logger().info(f"Capturing {demo_frames} demonstration frames")
            demo = []
            for pos in range(demo_frames):
                _, front, wrist, state = sensors.sample()
                demo.append(make_observation(front, wrist, state, instruction, position=pos, is_video_demo=True))
            memory = policy.observe_context(demo, memory)
        for step in range(steps):
            stamp, front, wrist, state = sensors.sample()
            obs = make_observation(front, wrist, state, instruction, position=demo_frames + step)
            predicted, memory = policy.act(obs, memory)
            chunk = as_numpy(predicted)
            if chunk.shape != (1, policy.action_horizon, 8):
                raise ValueError(f"Unexpected policy action shape {chunk.shape}")
            action = standard_joint_action(chunk[0, 0])
            age = (node.get_clock().now().nanoseconds - stamp) / 1e9
            print(json.dumps({"step": step, "stamp_ns": stamp, "age_s": age, "state": state.tolist(),
                              "predicted_action": action.tolist(), "execute": execute}), flush=True)
            if controllers:
                # Re-read joint feedback after inference before sending the first waypoint.
                _, _, _, fresh = sensors.sample()
                age = (node.get_clock().now().nanoseconds - stamp) / 1e9
                if not 0 <= age <= cfg["max_prediction_age"]:
                    raise RuntimeError(f"Prediction is stale ({age:.3f} s)")
                check_target(action, fresh[:7], cfg["max_joint_step"])
                controllers.execute(action)
                print(json.dumps({"step": step, "controller_result": "succeeded"}), flush=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
