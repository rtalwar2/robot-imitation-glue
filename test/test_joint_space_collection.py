"""tests for the action-space dispatch in the data collection loop, using the mock env (no hardware)."""

from types import SimpleNamespace

import numpy as np
import pytest

import robot_imitation_glue.collect_data as collect_data_module
from robot_imitation_glue.action_spaces import (
    abs_joints_to_policy_action,
    policy_action_to_abs_joints,
    resolve_action_type,
)
from robot_imitation_glue.base import ActionSpec, ActionType, BaseAgent, BaseDatasetRecorder
from robot_imitation_glue.collect_data import collect_data
from robot_imitation_glue.mock import MockActionSpec, MockEnv, mock_agent_to_pose_converter


class JointActionSpec(ActionSpec):
    action_type = ActionType.ABS_JOINT


class LoopMockEnv(MockEnv):
    """MockEnv that provides the image keys the collection loop visualizes and logs the env actions.

    It also allows to script keyboard events: `event_script` maps the index of a control cycle to the
    attribute of the `Event` object that should be set at the start of that cycle.
    """

    def __init__(self, event_holder, event_script):
        super().__init__()
        self.event_holder = event_holder
        self.event_script = event_script
        self.n_observations = 0
        self.eef_actions = []
        self.joint_actions = []

    def get_observations(self):
        attribute = self.event_script.get(self.n_observations)
        if attribute is not None:
            setattr(self.event_holder["event"], attribute, True)
        self.n_observations += 1

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        return {"scene_image": image, "wrist_image": image, "state": self.get_joint_configuration()}

    def act(self, robot_pose_se3, gripper_pose, timestamp):
        # note: MockEnv names this argument `gripper_opening`, the loops (and BaseEnv) use `gripper_pose`.
        self.eef_actions.append((robot_pose_se3, gripper_pose))
        MockEnv.act(self, robot_pose_se3, gripper_pose, timestamp)

    def act_joints(self, robot_joints, gripper_pose, timestamp):
        self.joint_actions.append((robot_joints, gripper_pose))
        MockEnv.act_joints(self, robot_joints, gripper_pose, timestamp)


class JointLoopMockEnv(LoopMockEnv):
    ACTION_SPEC = JointActionSpec


class ScriptedAgent(BaseAgent):
    """agent that replays a fixed list of actions (the last one is repeated)"""

    def __init__(self, action_spec, actions):
        self.ACTION_SPEC = action_spec
        self.actions = actions
        self.n_calls = 0

    def get_action(self, observation):
        del observation
        action = self.actions[min(self.n_calls, len(self.actions) - 1)]
        self.n_calls += 1
        return action


class ListDatasetRecorder(BaseDatasetRecorder):
    def __init__(self):
        self.episodes = []
        self.current_episode = None

    def start_episode(self):
        self.current_episode = []

    def record_step(self, obs, action):
        self.current_episode.append((obs, action))

    def save_episode(self):
        self.episodes.append(self.current_episode)
        self.current_episode = None

    @property
    def n_recorded_episodes(self):
        return len(self.episodes)


def mock_abs_pose_to_policy_action(robot_pose, gripper_pose, abs_se3_action, gripper_action):
    """absolute poses, encoded as [pos(3), rot-x(3), rot-y(3), gripper]"""
    del robot_pose, gripper_pose
    policy_action = np.zeros(10, dtype=np.float32)
    policy_action[:3] = abs_se3_action[:3, 3]
    policy_action[3:6] = abs_se3_action[:3, 0]
    policy_action[6:9] = abs_se3_action[:3, 1]
    policy_action[9] = gripper_action
    return policy_action


@pytest.fixture
def event_holder(monkeypatch):
    """run the collection loop headless: no keyboard listener and no rerun viewer"""
    holder = {}

    def fake_init_keyboard_listener(event, state):
        del state
        holder["event"] = event
        return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(collect_data_module, "init_keyboard_listener", fake_init_keyboard_listener)
    monkeypatch.setattr(collect_data_module.rr, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(collect_data_module.rr, "log", lambda *args, **kwargs: None)
    return holder


# record during cycles 0 and 1, stop recording in cycle 2 and quit in cycle 3.
EVENT_SCRIPT = {0: "start_recording", 2: "stop_recording", 3: "quit"}


def test_joint_space_collection(event_holder):
    env = JointLoopMockEnv(event_holder, EVENT_SCRIPT)
    env.reset()
    # a gello arm in joint mode provides absolute [joints(n), gripper] actions
    agent = ScriptedAgent(
        JointActionSpec,
        [np.array([10.0, 20.0, 0.5]), np.array([30.0, 40.0, 0.2]), np.array([50.0, 60.0, 0.0])],
    )
    recorder = ListDatasetRecorder()

    collect_data(
        env,
        agent,
        recorder,
        frequency=50,
        teleop_to_joints_converter=policy_action_to_abs_joints,
        abs_joints_to_policy_action=abs_joints_to_policy_action,
    )

    # the joint actions were sent to act_joints, the EEF path was not used at all
    assert env.eef_actions == []
    assert len(env.joint_actions) == 3
    np.testing.assert_allclose(env.get_joint_configuration(), [50, 60])
    assert env.get_gripper_opening() == 0.0

    # one episode with the two steps that were recorded, encoded as [joints(n), gripper]
    assert recorder.n_recorded_episodes == 1
    actions = [action for _, action in recorder.episodes[0]]
    assert len(actions) == 2
    for action in actions:
        assert action.dtype == np.float32
        assert action.shape == (3,)
    np.testing.assert_allclose(actions[0], [10.0, 20.0, 0.5])
    np.testing.assert_allclose(actions[1], [30.0, 40.0, 0.2])


def test_eef_collection_still_integrates_target_pose(event_holder):
    env = LoopMockEnv(event_holder, EVENT_SCRIPT)
    env.reset()
    initial_position = env.get_joint_configuration().copy()
    # the mock teleop converter adds the action to the current target pose
    agent = ScriptedAgent(MockActionSpec, [np.array([1.0, 0.0, 0, 0, 0, 0, 0], dtype=np.float32)])
    recorder = ListDatasetRecorder()

    collect_data(
        env,
        agent,
        recorder,
        frequency=50,
        teleop_to_pose_converter=mock_agent_to_pose_converter,
        abs_pose_to_policy_action=mock_abs_pose_to_policy_action,
    )

    # the EEF actions were sent to act, the joint path was not used at all
    assert env.joint_actions == []
    assert len(env.eef_actions) == 3
    # the target pose is integrated over the three control cycles
    np.testing.assert_allclose(env.get_joint_configuration(), initial_position + [3, 0])

    assert recorder.n_recorded_episodes == 1
    actions = [action for _, action in recorder.episodes[0]]
    assert len(actions) == 2
    np.testing.assert_allclose(actions[0][:2], initial_position + [1, 0])
    np.testing.assert_allclose(actions[1][:2], initial_position + [2, 0])


def test_missing_joint_converters_raises(event_holder):
    env = JointLoopMockEnv(event_holder, EVENT_SCRIPT)
    agent = ScriptedAgent(JointActionSpec, [np.zeros(3)])
    with pytest.raises(AssertionError):
        collect_data(env, agent, ListDatasetRecorder(), teleop_to_pose_converter=mock_agent_to_pose_converter)


def test_resolve_action_type():
    env = MockEnv()
    # both undeclared -> the EEF path, as before joint actions existed
    assert resolve_action_type(env, ScriptedAgent(None, [])) == ActionType.ABS_EEF
    assert resolve_action_type(env, ScriptedAgent(MockActionSpec, [])) == ActionType.ABS_EEF

    joint_env = JointLoopMockEnv({}, {})
    assert resolve_action_type(joint_env, ScriptedAgent(JointActionSpec, [])) == ActionType.ABS_JOINT
    # an agent that does not declare its action space is not checked
    assert resolve_action_type(joint_env, ScriptedAgent(None, [])) == ActionType.ABS_JOINT


def test_resolve_action_type_mismatch():
    with pytest.raises(ValueError) as error:
        resolve_action_type(JointLoopMockEnv({}, {}), ScriptedAgent(MockActionSpec, []))

    message = str(error.value)
    assert "JointLoopMockEnv uses abs_joint" in message
    assert "ScriptedAgent uses abs_eef" in message


def test_joint_policy_action_roundtrip():
    joints = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    gripper = 0.03

    policy_action = abs_joints_to_policy_action(np.zeros(6), 0.0, joints, gripper)
    assert policy_action.shape == (7,)
    assert policy_action.dtype == np.float32

    decoded_joints, decoded_gripper = policy_action_to_abs_joints(np.zeros(6), 0.0, policy_action)
    np.testing.assert_allclose(decoded_joints, joints, atol=1e-6)
    np.testing.assert_allclose(decoded_gripper, gripper, atol=1e-6)
