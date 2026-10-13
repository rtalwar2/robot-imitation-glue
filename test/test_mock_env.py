import numpy as np
import pytest

from robot_imitation_glue.base import ActionType, BaseEnv
from robot_imitation_glue.mock import MockEnv


def test_eef_action():
    env = MockEnv()
    env.reset()

    pose = np.eye(4)
    pose[:2, 3] = [10, 20]
    env.act(robot_pose_se3=pose, gripper_opening=0.05, timestamp=0)

    np.testing.assert_allclose(env.get_robot_pose_se3()[:2, 3], [10, 20])
    assert env.get_gripper_opening() == 0.05


def test_joint_action():
    env = MockEnv()
    env.reset()

    env.act_joints(robot_joints=np.array([30.0, 40.0]), gripper_pose=0.02, timestamp=0)

    np.testing.assert_allclose(env.get_joint_configuration(), [30, 40])
    np.testing.assert_allclose(env.get_robot_pose_se3()[:2, 3], [30, 40])
    assert env.get_gripper_opening() == 0.02


def test_action_spec():
    assert MockEnv.ACTION_SPEC.action_type == ActionType.ABS_EEF


def test_act_joints_raises_by_default():
    # envs that do not implement the joint path should raise instead of silently doing nothing
    with pytest.raises(NotImplementedError):
        BaseEnv.act_joints(MockEnv(), np.zeros(2), 0.0, 0)
