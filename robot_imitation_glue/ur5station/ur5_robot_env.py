"""UR5e teleoperation environment with a wrist RealSense camera and Gello4UR.

This variant is configured for setups without a parallel gripper.
"""

import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from transformers import ASTForAudioClassification

from robot_imitation_glue.hardware.ipc_btn import BTNSubscriber
from robot_imitation_glue.hardware.ipc_ft import FTSubscriber
from robot_imitation_glue.hardware.ipc_mic import SpectrogramSubscriberKaldi

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Prefer local workspace versions over globally visible packages from other projects.
_LOCAL_PATHS = [
    _REPO_ROOT / "airo-mono" / "airo-typing",
    _REPO_ROOT / "airo-mono" / "airo-spatial-algebra",
    _REPO_ROOT / "airo-mono" / "airo-robots",
    _REPO_ROOT / "airo-mono" / "airo-camera-toolkit",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-agents",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-devices",
]
for _path in _LOCAL_PATHS:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import cv2
import loguru
import numpy as np
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_spatial_algebra.se3 import SE3Container, normalize_so3_matrix
from ur_analytic_ik import ur5e

from airo_teleop_agents.gello_teleop_agents import Gello4UR
from airo_teleop_devices.gello_teleop_device import GelloTeleopDevice

from robot_imitation_glue.base import BaseEnv
from robot_imitation_glue.hardware.ipc_camera import RGBCameraPublisher, RGBCameraSubscriber, initialize_ipc
from airo_robots.grippers.hardware.schunk_process import SchunkGripperProcess

WRIST_CAM_RGB_TOPIC = "wrist_rgb"
WRIST_CAM_DEPTH_TOPIC = "wrist_depth"
WRIST_CAM_RESOLUTION_TOPIC = "wrist_resolution"

ROBOT_IP = "10.42.0.163"
GELLO_AGENT_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT792DZ5-if00-port0"
CAMERA_UPDATE_HZ = 30
logger = loguru.logger


class CameraFactory:
    @staticmethod
    def create_wrist_camera():
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense

        # 720p only, no fallback: touch_point_detector's radii are calibrated on 1280x720
        # frames, so a silent drop to 480p would skew touch-point detection for a whole
        # recording session. Fail loudly instead.
        logger.info(f"Starting RealSense at {Realsense.RESOLUTION_720} @ 30fps")
        return Realsense(resolution=Realsense.RESOLUTION_720, fps=30)

class UR5eStation(BaseEnv):

    def __init__(self,schunk, with_instrumentation=False,with_spectogram_model=False,use_internal_ft=True,with_spectogram=False):
        self.with_instrumentation = with_instrumentation
        self.with_spectogram_model = with_spectogram_model
        self.use_internal_ft = use_internal_ft
        # with_spectogram records the raw mel spectrogram as an observation (an input modality).
        # with_spectogram_model additionally runs a trained AST over it; the two are independent.
        self.with_spectogram = with_spectogram or with_spectogram_model
        logger.info("connecting to robot.")
        self.robot = URrtde(ROBOT_IP, URrtde.UR3E_CONFIG, gripper=None)

        logger.info("connecting to gripper.")
        self.gripper  = schunk

        initialize_ipc()
        logger.info("Creating wrist camera publisher.")
        self._wrist_camera_publisher = RGBCameraPublisher(
            CameraFactory.create_wrist_camera,
            WRIST_CAM_RGB_TOPIC,
            WRIST_CAM_DEPTH_TOPIC,
            WRIST_CAM_RESOLUTION_TOPIC,
            CAMERA_UPDATE_HZ,
        )
        self._wrist_camera_publisher.start()

        logger.info("Creating wrist camera subscriber.")
        self._wrist_camera_subscriber = RGBCameraSubscriber(
            WRIST_CAM_RESOLUTION_TOPIC,
            WRIST_CAM_RGB_TOPIC,
        )
        if not self.use_internal_ft:
            logger.info("creating FT subscriber")
            self.ft_subscriber = FTSubscriber("FT")

        if self.with_spectogram:
            logger.info("creating spectogram subscriber")
            self.spectogram_subscriber = SpectrogramSubscriberKaldi("KaldiSpectrogram")

        # if self.with_instrumentation:
        #     logger.info("creating button subscriber")
        #     self.button_subscriber = BTNSubscriber("BTN")

        # if self.with_spectogram_model:
        #     # Path to your saved model (usually the checkpoint folder with best metrics)
        #     # e.g., "./ast_delta_z/checkpoint-1900" or just "./ast_delta_z" if you saved the final model there
        #     MODEL_PATH = "/home/rtalwar/robot-imitation-glue/outputs/ramen-noodels/ast_delta_xyz_button_click_detector" 

        #     # MUST match the values printed during training
        #     self.TRAIN_MEAN = 0.6460392475170917
        #     self.TRAIN_STD  = 0.06145321258655634

        #     # Define labels (Update these to match your actual classes)
        #     ID2LABEL = {
        #         0: "Class_0",
        #         1: "Class_1"
        #     }

        #     # Select device
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     print(f"Inference running on: {self.device}")


        #     # ---------------------------------------------------------
        #     # 2. Load Model
        #     # ---------------------------------------------------------

        #     # The config saved in MODEL_PATH already contains the customized 
        #     # max_length (298) and architecture changes.
        #     self.model = ASTForAudioClassification.from_pretrained(MODEL_PATH)
        #     self.model.to(self.device)
        #     self.model.eval() # Set to evaluation mode (freezes Dropout, Batchnorm, etc.)

        #     print("Model loaded successfully.")

        time.sleep(2)

    def preprocess_spectrogram(self,spectrogram_array):
        """
        Prepares a raw numpy spectrogram for the AST model.
        Expected Input Shape: (Time, Freq) -> e.g., (298, 128)
        """
        # 1. Convert to numpy if not already
        spec = np.array(spectrogram_array, dtype=np.float32)

        # 2. Apply the EXACT same normalization as training
        # norm(x) = (x - mean) / (std * 2)
        spec_normalized = (spec - self.TRAIN_MEAN) / (self.TRAIN_STD * 2)

        # 3. Convert to PyTorch Tensor
        tensor = torch.tensor(spec_normalized)

        # 4. Add Batch Dimension if missing: (298, 128) -> (1, 298, 128)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            
        return tensor.to(self.device)

    def get_joint_configuration(self):
        return self.robot.get_joint_configuration()

    def get_robot_pose_euler(self):
        """
        pose as [x,y,z,rx,ry,rz] in robot base frame using Euler angles
        """
        hom_pose = self.robot.get_tcp_pose()
        rotation_vector = SE3Container.from_homogeneous_matrix(hom_pose).orientation_as_euler_angles
        position = hom_pose[:3, 3]
        return np.concatenate((position, rotation_vector), axis=0)

    def get_robot_pose_se3(self):
        return self.robot.get_tcp_pose()

    def move_robot_to_tcp_pose(self, pose,joint_speed):
        self.robot.move_to_tcp_pose(pose, joint_speed=joint_speed).wait()

    def move_gripper(self, width):
        # No parallel gripper in this setup.
        self.gripper.move(width=width)
        return

    def get_gripper_opening(self):
        return np.array([self.gripper.get_current_width()], dtype=np.float32)

    def get_camera_intrinsics(self):
        self._wrist_camera_subscriber._grab_images()
        return self._wrist_camera_subscriber.intrinsics_matrix() #also possible from rgb camera subscriber

    # def get_observations(self):
    #     start_time = time.time()
    #     wrist_image = self._wrist_camera_subscriber.get_rgb_image_as_int()
    #     robot_state = self.get_robot_pose_euler().astype(np.float32)
    #     joints = self.robot.get_joint_configuration().astype(np.float32)
    #     wrist_image_resized = cv2.resize(wrist_image, (320, 240), interpolation=cv2.INTER_CUBIC)

    #     obs_dict = {
    #         "wrist_image_original": wrist_image,
    #         "wrist_image": wrist_image_resized,
    #         "state": robot_state,
    #         "robot_pose": robot_state,
    #         "joints": joints,
    #     }
    #     logger.info(f"get_observations time: {time.time() - start_time}")

    #     return obs_dict
    def get_observations(self):

        wrist_image = self._wrist_camera_subscriber.get_rgb_image_as_int()
        if self.with_spectogram:
            spectogram_rgb_image, spectogram_values = self.spectogram_subscriber.get_spectogram()
        robot_state = self.get_robot_pose_euler().astype(np.float32)
        gripper_state = self.get_gripper_opening().astype(np.float32)
        joints = self.robot.get_joint_configuration().astype(np.float32)
        if self.use_internal_ft:
            ft = np.array(self.robot.rtde_receive.getActualTCPForce()).astype(np.float32)
        # else:
        #     ft = self.ft_subscriber.get_FT()

        # TODO: resize images (but still include the original?)

        # resize wrist images to 224x224
        # print(wrist_image.shape)
        # wrist_image_resized = cv2.resize(wrist_image, (224, 224))
        wrist_image_resized = cv2.resize(wrist_image, (320, 240))

        state = np.concatenate((joints, gripper_state), axis=0)
        obs_dict = {
            "wrist_image_original": wrist_image,
            "wrist_image": wrist_image_resized,
            "state": state,
            "robot_pose": robot_state,
            "gripper_state": gripper_state,
            "joints": joints,
            "ft": ft,
        }

        if self.with_spectogram:
            obs_dict["spectogram_image"] = spectogram_rgb_image
            obs_dict["spectogram_values"] = spectogram_values

        if self.with_instrumentation:
            button = self.button_subscriber.get_button_state()
            # print(button)

            obs_dict["btn_state"] = button
            state = np.concatenate((state, button), axis=0)

        if self.with_spectogram_model:
            inputs = self.preprocess_spectrogram(spectogram_values[:,:,0])
            with torch.no_grad():
                # Pass input_values specifically (AST expects this argument name)
                outputs = self.model(input_values=inputs)
                
                # Get Logits
                logits = outputs.logits
                
                # Convert to Probabilities (Softmax)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get Predicted Class
                pred_idx = torch.argmax(probs, dim=-1).item()
                obs_dict["pred_button_state"]=np.array([pred_idx],"float32")

        # obs_dict["state"] = state

        # print(f"the state type is {state.dtype}")
        for k in obs_dict.keys():
            obs_dict[k]=obs_dict[k].copy()

        return obs_dict
    def is_tcp_pose_reachable(self,new_pose):
        return self.robot.is_tcp_pose_reachable(new_pose)
    
    def act_tcp(self,new_pose,timestamp):
        current_time = time.time()
        duration = timestamp - current_time
        self.robot.servo_to_tcp_pose(new_pose, duration)
        return
    
    def act(self, robot_joints, timestamp):

        # normal joint space

        # move robot to target pose
        current_time = time.time()
        duration = timestamp - current_time
        if duration < 0:
            logger.warning("Action duration is negative, setting it to 0")
            duration = 0
        logger.debug(
            f"Moving robot to joint configuration {robot_joints} with duration {duration}"
        )

        # robot_pose_se3[:3, :3] = normalize_so3_matrix(robot_pose_se3[:3, :3])
        # self.robot.servo_to_tcp_pose(robot_pose_se3, duration)
        self.robot.servo_to_joint_configuration(robot_joints, duration)
        # move gripper to target width

        # do not wait, handling timings is the responsibility of the caller
        return

    # def act(self, robot_pose_se3, gripper_pose, timestamp):
    #     del gripper_pose

    #     # move robot to target pose
    #     current_time = time.time()
    #     duration = timestamp - current_time
    #     if duration < 0:
    #         logger.warning("Action duration is negative, setting it to 0")
    #         duration = 0
    #     logger.debug(f"Moving robot to pose \n {robot_pose_se3} with duration {duration}")

    #     robot_pose_se3[:3, :3] = normalize_so3_matrix(robot_pose_se3[:3, :3])

    #     z_coord = robot_pose_se3[2, 3]

    #     if z_coord < 0.0:
    #         # too far
    #         logger.warning("Z coordinate is below zero . not executing action")
    #         return

    #     valid_pose = True
    #     if not self.robot.is_tcp_pose_reachable(robot_pose_se3):
    #         logger.warning("TCP pose is not reachable, not executing action")
    #         valid_pose = False
    #     MAX_TRANSLATION = 0.15
    #     if np.linalg.norm(robot_pose_se3[:3, 3] - self.robot.get_tcp_pose()[:3, 3]) > MAX_TRANSLATION:
    #         logger.warning("TCP pose is too far from current pose, clippping translation.")
    #         # clip the translation.
    #         direction = robot_pose_se3[:3, 3] - self.robot.get_tcp_pose()[:3, 3]
    #         direction = direction / np.linalg.norm(direction)
    #         robot_pose_se3[:3, 3] = self.robot.get_tcp_pose()[:3, 3] + 0.5 * MAX_TRANSLATION * direction
    #         valid_pose = True

    #     if robot_pose_se3[2, 3] < 0.0:
    #         logger.warning("Z coordinate is below zero . not executing action")
    #         valid_pose = False

    #     # check if robot is still upright, by checking if the z-component of the z-vector is still negative.
    #     if robot_pose_se3[2, 2] > 0.0:
    #         logger.warning("robot gripper points upwards, not executing action.")
    #         valid_pose = False

    #     if valid_pose:
    #         self.robot.servo_to_tcp_pose(robot_pose_se3, duration)

    #     return

    def close(self):
        self._wrist_camera_publisher.stop()


def convert_abs_gello_actions_to_se3(action: np.ndarray):
    tcp_pose = np.eye(4)
    joints = action[:6]
    pose = ur5e.forward_kinematics_with_tcp(*joints, tcp_pose)
    return pose


if __name__ == "__main__":
    env = UR5eStation()

    gello_config = deepcopy(GelloTeleopDevice.GELLO1_DEFAULT_CONFIG)
    agent = Gello4UR(
        gello_usb_port=GELLO_AGENT_PORT,
        gello_config=gello_config,
        ur_robot=env.robot,
        use_joint_space=True,
    )

    cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wrist", 640, 480)

    try:
        input("Press Enter to start teleoperation")
        action = agent.get_action()
        # robot_se3 = convert_abs_gello_actions_to_se3(action)
        env.robot.servo_to_joint_configuration(action, 5.0).wait()

        while True:
            loop_time = time.time()
            obs = env.get_observations()
            cv2.imshow("wrist", obs["wrist_image"])

            action = agent.get_action()
            # robot_se3 = convert_abs_gello_actions_to_se3(action)
            env.act(robot_joints=action, timestamp=time.time() + 0.1)
            print(f"Action: {action}")
            print(f"Robot joints: {env.get_joint_configuration()}")
            print("difference:", env.get_joint_configuration() - action)
            loop_duration = time.time() - loop_time
            key = cv2.waitKey(max(1, int(100 - loop_duration * 1000)))
            if key == ord("q"):
                break
    finally:
        env.close()
        cv2.destroyAllWindows()

