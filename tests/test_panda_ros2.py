"""Hardware-independent checks for the Panda ROS boundary."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vla_ttt import panda_ros2 as panda
from vla_ttt.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs/inference/panda_ros2.yaml"
Q = np.array([0., 0., 0., -1.5, 0., 1.5, 0.5], dtype=np.float32)


class PandaTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)["panda"]
        panda.validate_settings(self.cfg)

    def test_joint_order_and_padded_bgr(self):
        sensors = panda.Sensors.__new__(panda.Sensors)
        sensors.cfg, sensors.latest = self.cfg, {}
        names = self.cfg["arm_joints"]
        msg = NS(header=NS(stamp=NS(sec=1, nanosec=0)), name=names[::-1], position=Q[::-1])
        sensors.joints(self.cfg["joint_states"], msg)
        np.testing.assert_array_equal(sensors.latest["arm_joints"][1], Q)
        img = NS(encoding="bgr8", width=1, height=2, step=4, data=bytes(range(8)))
        np.testing.assert_equal(panda.rgb_image(img), [[[2, 1, 0]], [[6, 5, 4]]])

    def test_outside_limits_and_large_steps(self):
        action = np.r_[Q, 1.]
        panda.check_target(action, Q, .05)
        with self.assertRaisesRegex(ValueError, "max_joint_step"):
            panda.check_target(action, Q + .1, .05)
        action[3] = 0
        with self.assertRaisesRegex(ValueError, "joint limits"):
            panda.check_target(action, Q, .05)

    def rollout(self, execute, age=0.1):
        ros = MagicMock()
        node = ros.create_node.return_value
        node.get_clock.return_value.now.return_value.nanoseconds = int((1 + age) * 1e9)
        modules = {"rclpy": ros, "rclpy.signals": NS(SignalHandlerOptions=NS(NO=0))}
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        sample = (10**9, rgb, rgb, np.r_[Q, .02])
        policy = MagicMock(action_horizon=40)
        policy.observe_context.return_value = "memory after demo"
        chunk = np.tile(np.r_[Q, -1], (1, 40, 1))
        policy.act.return_value = chunk, "next memory"
        with patch.dict(sys.modules, modules), patch.object(panda, "Sensors") as sensors, \
                patch.object(panda, "Controllers") as controllers:
            sensors.return_value.sample.return_value = sample
            panda.run(policy, self.cfg, "pick", steps=1, demo_frames=2, execute=execute)
        self.assertTrue(all(obs["is_video_demo"] for obs in policy.observe_context.call_args.args[0]))
        self.assertEqual(policy.act.call_args.args[1], "memory after demo")
        return controllers

    def test_shadow_and_live(self):
        self.rollout(False).assert_not_called()
        controller = self.rollout(True)
        np.testing.assert_equal(controller.return_value.execute.call_args.args[0], np.r_[Q, -1])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.rollout(True, age=5)

    def test_controller_payload_and_gripper_changes(self):
        control = panda.Controllers.__new__(panda.Controllers)
        control.cfg, control.last_open = self.cfg, None
        control.arm, control.hand = object(), object()
        control.send = MagicMock(return_value=NS(error_code=0, SUCCESSFUL=0, reached_goal=True))
        arm_type = NS(Goal=lambda: NS(trajectory=NS()))
        hand_type = NS(Goal=lambda: NS(command=NS()))
        modules = {"control_msgs.action": NS(FollowJointTrajectory=arm_type, GripperCommand=hand_type),
                   "rclpy.duration": NS(Duration=lambda seconds: NS(to_msg=lambda: seconds)),
                   "trajectory_msgs.msg": NS(JointTrajectoryPoint=NS)}
        with patch.dict(sys.modules, modules):
            for grip in [-1, -1, 1]:
                control.execute(np.r_[Q, grip])
        calls = control.send.call_args_list
        self.assertEqual([c.args[0] for c in calls], [control.arm, control.hand, control.arm, control.arm, control.hand])
        self.assertEqual(calls[0].args[1].trajectory.joint_names, self.cfg["arm_joints"])
        np.testing.assert_equal(calls[0].args[1].trajectory.points[0].positions, Q)
        self.assertEqual(calls[1].args[1].command.position, self.cfg["close_finger_position"])
        self.assertEqual(calls[4].args[1].command.position, self.cfg["open_finger_position"])
