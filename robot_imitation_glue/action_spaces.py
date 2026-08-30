"""Helpers to deal with the action space (EEF vs joint) of an env/agent pair.

Joint-space actions are stored for the policy/dataset as a flat vector

    [joints(n), gripper]

i.e. the absolute target joint configuration followed by the absolute target gripper width. This
mirrors the `[pos(3), rot-x(3), rot-y(3), gripper]` encoding that is used for absolute EEF actions
(see `ur5station/data_collection.py`).
"""

import numpy as np

from robot_imitation_glue.base import ActionType

JOINT_ACTION_TYPES = (ActionType.ABS_JOINT, ActionType.REL_JOINT)


def is_joint_action_type(action_type: ActionType) -> bool:
    """whether actions of this type should be sent to `env.act_joints` instead of `env.act`"""
    return action_type in JOINT_ACTION_TYPES


def resolve_action_type(env, *agents) -> ActionType:
    """check that the env and the agent(s) agree on the action space and return that action space.

    Envs/agents that do not declare an action type (ACTION_SPEC is None, which is still the case for
    e.g. the UR5 station and the gello/spacemouse agents) are considered 'unspecified' and are not
    checked. If nobody declares an action type, we default to ABS_EEF, which is what all scripts did
    before joint actions existed.
    """
    declarations = {
        type(component).__name__: component.ACTION_SPEC.action_type
        for component in (env, *agents)
        if getattr(component.ACTION_SPEC, "action_type", None) is not None
    }

    if len(set(declarations.values())) > 1:
        raise ValueError(
            "action space mismatch: "
            + ", ".join(f"{name} uses {action_type.value}" for name, action_type in declarations.items())
        )

    return next(iter(declarations.values()), ActionType.ABS_EEF)


def abs_joints_to_policy_action(robot_joints, gripper_width, abs_joints_action, gripper_action) -> np.ndarray:
    """absolute joint targets, encoded as [joints(n), gripper]"""
    del robot_joints, gripper_width  # the encoding is absolute, the current state is not needed
    policy_action = np.concatenate([np.asarray(abs_joints_action).ravel(), np.asarray(gripper_action).ravel()])
    return policy_action.astype(np.float32)


def policy_action_to_abs_joints(robot_joints, gripper_width, policy_action):
    """decode a [joints(n), gripper] action into an absolute joint configuration and gripper width.

    This is the inverse of `abs_joints_to_policy_action`. It can also be used as teleop converter for
    teleop devices that already provide absolute joint configurations, such as a Gello arm.
    """
    del robot_joints, gripper_width
    policy_action = np.asarray(policy_action, dtype=np.float64)
    return policy_action[:-1], policy_action[-1]
