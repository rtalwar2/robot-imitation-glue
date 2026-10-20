"""Move ur_right to its neutral joint configuration (RIGHT_NEUTRAL_JOINTS) at RIGHT_JOINT_SPEED.

Only the right arm moves; the left arm and its gripper are not touched. The move is a plain joint move with no
collision checking, so make sure ur_left is clear of the bottle (e.g. at LEFT_HOME_JOINTS) first.

Usage (from bottle_experiment/):
    python move_right_to_neutral.py
"""

import numpy as np
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from calibrate_and_hover_bottle import RIGHT_ROBOT_IP
from open_bottle_agent import RIGHT_JOINT_SPEED, RIGHT_NEUTRAL_JOINTS


def main():
    ur_right = URrtde(RIGHT_ROBOT_IP, URrtde.UR3E_CONFIG, gripper=None)
    print("current joints", np.round(ur_right.get_joint_configuration(), 3))
    input(f"Press Enter to move the RIGHT arm to neutral {RIGHT_NEUTRAL_JOINTS} (Ctrl+C to abort)...")
    ur_right.move_to_joint_configuration(np.array(RIGHT_NEUTRAL_JOINTS), joint_speed=RIGHT_JOINT_SPEED).wait()
    print("reached joints", np.round(ur_right.get_joint_configuration(), 3))


if __name__ == "__main__":
    main()
