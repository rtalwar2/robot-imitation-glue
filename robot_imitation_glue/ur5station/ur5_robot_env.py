"""
Environment for the UR5 station: a ZED scene camera, a RealSense wrist camera and a UR5e robot with a Schunk EGK40 gripper
"""

import os
import time
from copy import deepcopy

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.cameras.zed.zed import Zed
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_teleop_devices.gello_teleop_device import GelloTeleopDevice
from ur_analytic_ik import ur5e

from robot_imitation_glue.agents.airo_teleop_agent import make_gello_agent
from robot_imitation_glue.hardware.grippers.schunk_process import SchunkGripperProcess

# the TCP offset is also needed by the offline dataset preparation (IK), so it is defined in the shared
# preprocessing module to keep the online and offline kinematics in sync.
from robot_imitation_glue.preprocessing import SCHUNK_TCP_OFFSET
from robot_imitation_glue.station_env import ManipulatorStation

WRIST_REALSENSE_SERIAL = "817612070315"
SCENE_ZED_SERIAL = "31733653"

ROBOT_IP = "10.42.0.163"

SCHUNK_GRIPPER_HOST = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0,11,115200,8E1"  # run bks_scan -H <usb> to find the slaveID, run dmesg | grep tty to find the usb port

HOME_JOINTS = np.array([-180, -90, 90, -90, -90, -90]) * np.pi / 180  # for left UR5e on dual arm setup in mano lab.

GELLO_AGENT_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT792DZ5-if00-port0"
logger = loguru.logger


class CameraFactory:
    def create_wrist_camera():
        return Realsense(resolution=Realsense.RESOLUTION_480, fps=30, serial_number=WRIST_REALSENSE_SERIAL)

    def create_scene_camera():
        return Zed(
            resolution=Zed.RESOLUTION_VGA, fps=30, depth_mode=Zed.NONE_DEPTH_MODE, serial_number=SCENE_ZED_SERIAL
        )


class UR5eStation(ManipulatorStation):
    MAX_TRANSLATION = 0.15

    def __init__(self):
        # set environment variable for bks gripper comm
        os.environ["BKS_HOST"] = SCHUNK_GRIPPER_HOST
        gripper = SchunkGripperProcess(SCHUNK_GRIPPER_HOST)
        time.sleep(2)
        gripper.max_grasp_force = gripper.gripper_specs.min_force  # minimal force for EGK40 is 55N
        gripper.speed = gripper.gripper_specs.max_speed
        logger.info("connecting to robot.")
        robot = URrtde(ROBOT_IP, URrtde.MANIPULATOR_SPECS[URrtde.URModels.UR5e], gripper=gripper)

        input("Press Enter to move the robot to its home configuration (HOME_JOINTS)...")
        robot_awaitable = robot.move_to_joint_configuration(HOME_JOINTS)  # do not wait, let cameras initialize first
        super().__init__(robot, CameraFactory.create_wrist_camera, CameraFactory.create_scene_camera)
        robot_awaitable.wait()

    def preprocess_images(self, wrist_image, scene_image):
        wrist_image_resized = cv2.resize(wrist_image, (320, 240), interpolation=cv2.INTER_CUBIC)
        scene_image_resized = cv2.resize(scene_image, (320, 240), interpolation=cv2.INTER_CUBIC)
        return wrist_image_resized, scene_image_resized

    def is_safe_pose(self, robot_pose_se3):
        if robot_pose_se3[2, 3] < 0.0:
            logger.warning("Z coordinate is below zero, not executing action.")
            return False

        # clip the translation of targets that are too far from the current pose
        current_position = self.robot.get_tcp_pose()[:3, 3]
        translation = robot_pose_se3[:3, 3] - current_position
        if np.linalg.norm(translation) > self.MAX_TRANSLATION:
            logger.warning("TCP pose is too far from current pose, clipping translation.")
            direction = translation / np.linalg.norm(translation)
            robot_pose_se3[:3, 3] = current_position + 0.5 * self.MAX_TRANSLATION * direction

        if not self.robot.is_tcp_pose_reachable(robot_pose_se3):
            logger.warning("TCP pose is not reachable, not executing action.")
            return False

        # check if robot is still upright, by checking if the z-component of the z-vector is still negative.
        if robot_pose_se3[2, 2] > 0.0:
            logger.warning("robot gripper points upwards, not executing action.")
            return False
        return True

    def servo_gripper(self, width):
        self.gripper.servo(width)

    def close(self):
        super().close()
        self.gripper.shutdown()


def abs_joint_policy_action_to_se3(current_pose, current_gripper_state, action: np.ndarray):
    del current_pose, current_gripper_state
    joints = action[:6]
    gripper = action[6]
    tcp_pose = np.eye(4)
    tcp_pose[2, 3] = SCHUNK_TCP_OFFSET
    pose = ur5e.forward_kinematics_with_tcp(*joints, tcp_pose)
    return pose, gripper


# the Gello agent returns [6 joints, gripper width in m], the same format as the joint policy actions
convert_abs_gello_actions_to_se3 = abs_joint_policy_action_to_se3

# the station's Gello arm is the one labeled "Gello1", its calibration lives in airo-teleop.
# copied, because the Gello agent writes the robot's start joints into the config.
# to recalibrate: airo-teleop-devices/calibration/gello_calibration.py
GELLO_CONFIG = deepcopy(GelloTeleopDevice.GELLO1_DEFAULT_CONFIG)


if __name__ == "__main__":
    env = UR5eStation()
    agent = make_gello_agent(env, GELLO_AGENT_PORT, GELLO_CONFIG)

    cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wrist", 640, 480)
    cv2.resizeWindow("scene", 640, 480)

    input("Press Enter to start teleoperation")
    obs = env.get_observations()
    action = agent.get_action(obs)
    robot_se3, gripper = convert_abs_gello_actions_to_se3(env.get_robot_pose_se3(), env.get_gripper_opening(), action)
    # first move the robot slowly to the current teleop pose
    env.move_robot_to_tcp_pose(robot_se3)

    print("teleoperating, press q in an image window to quit")
    while True:
        loop_time = time.time()
        obs = env.get_observations()
        print(time.time())
        print(obs["wrist_image"].shape)
        cv2.imshow("wrist", obs["wrist_image"])
        cv2.imshow("scene", obs["scene_image"])

        action = agent.get_action(obs)
        robot_se3, gripper = convert_abs_gello_actions_to_se3(
            env.get_robot_pose_se3(), env.get_gripper_opening(), action
        )
        env.act(robot_pose_se3=robot_se3, gripper_pose=gripper, timestamp=time.time() + 0.1)
        print(obs["state"])

        loop_duration = time.time() - loop_time
        # wait for 100ms - loop time
        key = cv2.waitKey(max(1, int(100 - loop_duration * 1000)))
        if key == ord("q"):
            break

    agent.close()
    env.close()
