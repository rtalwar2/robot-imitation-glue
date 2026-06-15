import time

# create type for callable that takes obs and returns action
from typing import Callable , Any, List, Dict, Optional, Union, Tuple


from airo_typing import HomogeneousMatrixType, JointConfigurationType, NumpyDepthMapType, NumpyIntImageType
from airo_spatial_algebra import SE3Container, transform_points
import cv2
import loguru
import numpy as np
import rerun as rr

from robot_imitation_glue.base import BaseAgent, BaseDatasetRecorder, BaseEnv
from robot_imitation_glue.button_detector.ButtonDetector import ButtonDetector
from robot_imitation_glue.forward_kinematics_helper import forward_kinematics_ur3e
from robot_imitation_glue.utils import precise_wait
converter_callable = Callable[dict[str, np.ndarray], np.ndarray]

logger = loguru.logger


class State:
    is_recording = False
    is_stopped = False
    is_paused = False
    go_down=True
    phase1 = False
    phase2 = False
    phase3 = False
class Event:
    start_recording = False
    stop_recording = False
    delete_last = False
    pause = False
    resume = False
    quit = False
    go_up=False

    def clear(self):
        for attr in self.__dict__:
            setattr(self, attr, False)


def init_keyboard_listener(event: Event, state: State):
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            # "space bar"
            if key == keyboard.Key.enter and not state.is_recording:
                event.start_recording = True

            # elif key == keyboard.Key.enter and state.is_recording and state.go_down:
            #     event.go_up = True

            elif key == keyboard.Key.enter and state.is_recording:
                event.stop_recording = True
                event.go_up = False


            elif hasattr(key, "char") and key.char == "p" and not state.is_recording and not state.is_paused:
                # pause the episode
                event.pause = True

            elif hasattr(key, "char") and key.char == "p" and state.is_paused:
                # resume the episode
                event.resume = True

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and state.is_recording:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


def collect_data(  # noqa: C901
    env: BaseEnv,
    teleop_agent: BaseAgent,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
    teleop_to_pose_converter: converter_callable = None,
    abs_pose_to_policy_action: converter_callable = None,
):
    assert env.ACTION_SPEC == teleop_agent.ACTION_SPEC
    rr.init("robot_imitation_glue", spawn=True)
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency

    observation = env.get_observations()
    action = teleop_agent.get_action(observation)
    env.act(
        robot_joints=action[0:6],
        gripper_pose=0,
        timestamp=time.time() + 5,
    )
    # --- Configuration for Adaptive Speed ---
    BUTTON_Z = 0.036      # The height of the button
    MIN_STEP = 0.001    # Minimum movement step (when very close)
    MAX_STEP = 0.01     # Maximum movement step (when far away)
    SPEED_GAIN = 0.3     # How quickly speed increases with distance (higher = faster acceleration)
    # ----------------------------------------

    while not state.is_stopped:
        cycle_end_time = time.time() + control_period

        before_observation_time = time.time()
        observation = env.get_observations()
        after_observation_time = time.time()
        observation_time= after_observation_time - before_observation_time
        # print("observation time: ", observation_time)

        # update & handle state machine events
        if not state.is_recording and event.start_recording:
            state.is_recording = True
            print("start recording")
            dataset_recorder.start_episode()
            # fixed_action = np.array([0.005], dtype=np.float64) #add random noise
            
        elif state.is_recording and event.go_up:
            # fixed_action = np.array([-0.005], dtype=np.float64) #add random noise
            state.go_down=False

        elif state.is_recording and event.stop_recording:
            state.is_recording = False
            print("stop recording")
            # save episode
            dataset_recorder.save_episode()
            input("hold teleop in place now!")
            observation = env.get_observations()
            action = teleop_agent.get_action(observation)
            env.act(
                robot_joints=action[0:6],
                gripper_pose=0,
                timestamp=time.time() + 5,
            )
            state.go_down=True

            # TODO: allow for textual description of the episode?

        elif event.delete_last and not state.is_recording:
            print("delete last episode")
            state.is_recording=False
            # delete last episode
            dataset_recorder.delete_episode()
            
        elif event.pause and not state.is_recording:
            state.is_paused = True
            print("pause teleop")

        elif event.resume and state.is_paused:
            state.is_paused = False
            print("resume teleop")

        elif event.quit:
            print("quit")
            state.is_stopped = True
            listener.stop()
            dataset_recorder.finish_recording()
            return

        # clear all events
        event.clear()

        # update GUI.
        vis_img = observation["wrist_image"].copy()
        if observation["btn_state"]==0:
            # event.go_up
            # fixed_action = np.array([-0.005], dtype=np.float64) #add random noise
            state.go_down=False
        # visualize state is_recording, is_paused
        if state.is_recording:
            cv2.putText(vis_img, "RECORDING", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if state.is_paused:
            cv2.putText(vis_img, "PAUSED", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(
            vis_img,
            f" # episodes: {dataset_recorder.n_recorded_episodes}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        rr.log("wrist_image", rr.Image(vis_img, rr.ColorModel.RGB))
        rr.log("wrist_image original", rr.Image(observation["wrist_image_original"], rr.ColorModel.RGB))
        rr.log("spectogram", rr.Image(observation["spectogram_image"], rr.ColorModel.RGB))
        rr.log("spectogramgray", rr.Image(observation["spectogram_values"], rr.ColorModel.RGB))
        # rr.log("spectogrambgr", rr.Image(observation["spectogram_image"], rr.ColorModel.BGR))
        # print(f"wrist image shape: {observation['wrist_image'].shape}")
        # print(f"spectogram image shape: {observation['spectogram_image'].shape}")
        # rr.log("scene_image", rr.Image(observation["scene_image"], rr.ColorModel.RGB))
        rr.log("joints", rr.TextLog(str(observation["joints"])))
        rr.log("btn_state", rr.Scalar(float(observation["btn_state"])))
        # if paused, do not collect teleop or execute action
        if state.is_paused:
            time.sleep(0.1)
            continue

        if state.is_recording:
            # --- ADAPTIVE ACTION CALCULATION ---
            # 1. Get current Z height
            current_z = env.get_robot_pose_se3()[2,3]
            
            # 2. Calculate distance to button
            # We use abs() to ensure positive distance even if we overshoot slightly below 0.04
            distance = abs(current_z - BUTTON_Z)
            
            # 3. Calculate step size (Proportional Control)
            # Speed = Gain * Distance
            step_size = distance * SPEED_GAIN
            
            # 4. Clip step size to ensure we don't stall (min) or move too dangerously fast (max)
            step_size = np.clip(step_size, MIN_STEP, MAX_STEP)
            
            # 5. Apply Direction
            if state.go_down:
                current_action = np.array([step_size], dtype=np.float64)
            else:
                current_action = np.array([-step_size], dtype=np.float64)
            
            # Log the speed for debugging
            print(f"Z: {current_z}, Dist: {distance}, Action: {current_action[0]}")

            env.act_deltaz(current_action, env.get_robot_pose_se3(), time.time() + control_period)
            
            # Record the adaptive action
            dataset_recorder.record_step(observation, current_action)
            
        else:
            # use teleop actions
            action = teleop_agent.get_action(observation)
            logger.info(f"Action: {action}")

            gripper_target = 0
            env.act(
                robot_joints=action[0:6],
                gripper_pose=gripper_target,
                timestamp=time.time() + control_period,
            )


        # wait for end of the control period
        if cycle_end_time > time.time():
            precise_wait(cycle_end_time)
        else:
            print("cycle time exceeded control period")

        # update the target pose and target gripper state for the next iteration
        # target_pose = new_robot_target_se3_pose
        # target_gripper_state = new_gripper_target_width

        # TODO: we now use 'integration' to get the next target pose instead of using the current pose.
        # this is to avoid 'shaking' of the robot, as is done in diffusion policy teleop for example.
        # but need to verify that this does not causepp mismatch between teleop and policy.
        # and should also check if the distance between the target and the actual robot does not diverge too much.

def randomize_initial_pose(X_B_TCP_touch, 
                           xy_range=0.05, 
                           z_range=0.05):
    """
    Creates a randomized initial TCP pose (rotation + translation)
    and ensures the pose is reachable. If not, it keeps retrying.
    """
    # --- Random rotation around Z axis ---
    theta = np.random.uniform(-np.pi,0)

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,               0,            1, 0],
        [0,               0,            0, 1],
    ])

    X_rotated = X_B_TCP_touch @ Rz

    # --- Random XYZ offset in BASE frame ---
    dx = np.random.uniform(-xy_range, xy_range)
    dy = np.random.uniform(-xy_range, xy_range)
    dz = np.random.uniform(0, z_range)

    offset = np.array([dx, dy, dz])
    X_rotated[:3, 3] += offset
    return X_rotated

def create_action(tcp_pose,button_position):
    actions = button_position - tcp_pose[:3,3]
    actions_in_toolspace = np.linalg.inv(tcp_pose[:3,:3])@actions # action is now in tool space
    return actions_in_toolspace


def create_action_from_tool_offset(tcp_pose, target_position, tool_offset):
    # Convert a point fixed in the tool frame (e.g. gripper edge) into base frame,
    # then compute the delta needed to move that point to the target.
    edge_position_base = transform_points(tcp_pose, tool_offset)
    actions = target_position - edge_position_base
    actions_in_toolspace = np.linalg.inv(tcp_pose[:3, :3]) @ actions
    return actions_in_toolspace


def rotation_matrix_to_6d(rotation_matrix):
    # Rotation 6D representation uses the first two columns of the rotation matrix.
    return np.concatenate((rotation_matrix[:, 0], rotation_matrix[:, 1]), axis=0)


def step_action_to_policy_action_6d(step_action):
    # Policy action format: [delta_xyz(3), rotation_6d(6)]
    if step_action.shape[0] != 6:
        raise ValueError(f"Expected step action with 6 elements, got shape {step_action.shape}")

    R_delta = SE3Container.from_rotation_vector_and_translation(step_action[3:6], np.zeros(3)).rotation_matrix

    return np.concatenate((step_action[:3], rotation_matrix_to_6d(R_delta)), axis=0).astype(np.float64)


def policy_action_to_tcp_pose_old(tcp_pose, policy_action, eps=1e-8):
    # Policy action format: [delta_xyz(3), rotation_6d(6)] where rotation is a relative delta.
    if policy_action.shape[0] != 9:
        raise ValueError(f"Expected policy_action with 9 elements, got shape {policy_action.shape}")

    delta_xyz_tool = policy_action[:3]
    rot6d = policy_action[3:9]

    x = rot6d[:3]
    x = x / np.linalg.norm(x)
    y = rot6d[3:] - np.dot(rot6d[3:], x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    R_delta = np.column_stack((x, y, z))

    final_pose = tcp_pose.copy()
    delta_xyz_base = tcp_pose[:3, :3] @ delta_xyz_tool
    final_pose[:3, 3] += delta_xyz_base
    final_pose[:3, :3] = tcp_pose[:3, :3] @ R_delta
    return final_pose


def policy_action_to_tcp_pose(tcp_pose, policy_action, eps=1e-8):
    # If tcp_pose is a spatialmath.SE3 object, extract its underlying 4x4 array
    if hasattr(tcp_pose, "A"):
        tcp_pose = tcp_pose.A

    # Enforce double precision (float64) to prevent precision-loss validation failures
    tcp_pose = np.asarray(tcp_pose, dtype=np.float64)
    policy_action = np.asarray(policy_action, dtype=np.float64)

    if policy_action.shape[0] != 9:
        raise ValueError(f"Expected policy_action with 9 elements, got shape {policy_action.shape}")

    delta_xyz_tool = policy_action[:3]
    rot6d = policy_action[3:9]

    # --- Robust Gram-Schmidt Orthogonalization for R_delta ---
    x = rot6d[:3]
    x_norm = np.linalg.norm(x)
    x = x / x_norm if x_norm > eps else np.array([1.0, 0.0, 0.0])

    y = rot6d[3:] - np.dot(rot6d[3:], x) * x
    y_norm = np.linalg.norm(y)
    if y_norm > eps:
        y = y / y_norm
    else:
        # Safe fallback if x and y are parallel
        y = np.array([0.0, 1.0, 0.0]) if abs(x[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        y = y - np.dot(y, x) * x
        y = y / np.linalg.norm(y)

    z = np.cross(x, y)
    R_delta = np.column_stack((x, y, z))

    # --- Raw composition ---
    R_final_raw = tcp_pose[:3, :3] @ R_delta

    # --- SVD Orthonormalization (Project back onto SO(3)) ---
    # This strips away numerical drift, forcing the rotation to be mathematically perfect
    U, _, Vt = np.linalg.svd(R_final_raw)
    R_final_valid = U @ Vt
    
    # Ensure a right-handed system (prevents unwanted reflections)
    if np.linalg.det(R_final_valid) < 0:
        U[:, -1] *= -1
        R_final_valid = U @ Vt

    # --- Construct the Homogeneous Matrix ---
    # Initialize clean SE(3) identity template to ensure the bottom row is exactly [0, 0, 0, 1]
    final_pose = np.eye(4, dtype=np.float64)
    
    # Calculate updated translation vector
    delta_xyz_base = tcp_pose[:3, :3] @ delta_xyz_tool
    final_pose[:3, 3] = tcp_pose[:3, 3] + delta_xyz_base
    
    # Assign the mathematically sanitized rotation matrix
    final_pose[:3, :3] = R_final_valid

    return final_pose

def action_to_tcp_pose(tcp_pose,action):
    # Backward-compatible wrapper: convert delta action (xyz + optional rotvec)
    # into policy action (xyz + rotation_6d), then use the policy converter.
    if action.shape[0] == 3:
        action_6d = np.concatenate((action, np.zeros(3)), axis=0)
    elif action.shape[0] == 6:
        action_6d = action
    else:
        raise ValueError(f"Expected action with 3 or 6 elements, got shape {action.shape}")

    policy_action = step_action_to_policy_action_6d(action_6d)
    return policy_action_to_tcp_pose(tcp_pose, policy_action)

def collect_data_xyz_white_switch(  # noqa: C901
    env: BaseEnv,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
    detected_3d_button_positions = None,
    pre_touching_pose: HomogeneousMatrixType = None,
    deterministic_poses:list[HomogeneousMatrixType]=None,
):
    # Local import avoids top-level circular imports: eval_agent_delta_z imports from this module.
    from robot_imitation_glue.eval_agent_delta_z import generate_deterministic_poses

    rr.init("robot_imitation_glue")
    rr.spawn(memory_limit="5GB")
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency

    if len(deterministic_poses) == 0:
        raise ValueError("No deterministic reachable poses found, please shift the button")

    def move_to_start_pose_with_negative_z_detour(target_pose):
        """
        Move to target start pose with a deterministic orientation policy:
        - If the target requires a positive yaw turn larger than 90 deg from current,
          first rotate -90 deg around tool Z at the current position.
        - Otherwise, move directly to target pose.
        """
        R1 = pre_touching_pose[:3, :3]
        R2 = target_pose[:3, :3]
        R_diff = R1.T @ R2
        angle = np.arccos((np.trace(R_diff) - 1) / 2)
        print("angle between R1 and R2:", np.degrees(angle), "degrees")
        if np.degrees(angle)>90:
            print("applying detour because angle is greater than 90 degrees")
            Rz_minus_90 = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
            detour_pose = target_pose.copy()
            detour_pose[:3, :3] = Rz_minus_90

            if env.is_tcp_pose_reachable(detour_pose):
                print("Applying negative-Z yaw detour before start pose")
                env.move_robot_to_tcp_pose(detour_pose)

        env.move_robot_to_tcp_pose(target_pose)

    pose_idx = dataset_recorder.n_recorded_episodes % len(deterministic_poses)
    new_pose = deterministic_poses[pose_idx]
    move_to_start_pose_with_negative_z_detour(new_pose)

    # observation = env.get_observations()
    # image = observation["wrist_image_original"]
    # BUTTON_Z = 0.036 
    # button_position = button_detector.get_button_position_world_height_known(image,BUTTON_Z,pre_touching_pose)

    # --- Configuration for Adaptive Speed ---
    target_position = detected_3d_button_positions 
    # Simulation Constants adapted for Robot Safety
    BUTTON_PRE_Z_OFFSET = 0.015 # 1cm above button
    LATERAL_TOLERANCE   = 0.002 # 1mm (When we consider X/Y aligned)
    MIN_STEP = 0.002  # 1mm min step
    MAX_STEP = 0.03   # 1cm max step (Robot safety cap)
    #best values from sim
    # MIN_STEP = 0.00001
    # MAX_STEP = 0.04
    BASE_SPEED_GAIN = 0.1
    BASE_LATERAL_GAIN = 0.1
    BASE_ROT_GAIN = 0.1
    RETRACT_HEIGHT_OFFSET = 0.10 # 10cm above button
    GRIPPER_EDGE_OFFSET_TOOL = np.array([0.0, -0.0152, 0.0])
    ORIENTATION_TOLERANCE = np.deg2rad(5.0)
    MIN_ROT_STEP = np.deg2rad(0.5)
    MAX_ROT_STEP = np.deg2rad(3.0)
    # ----------------------------------------

    # styles = np.logspace(np.log10(0.1), np.log10(10.0), 10)
    # style_idx=0

    k_xy = BASE_LATERAL_GAIN 
    k_z  = BASE_SPEED_GAIN
    k_rot = BASE_ROT_GAIN

    event.start_recording=True
    try:

        while not state.is_stopped:
            cycle_end_time = time.time() + control_period

            before_observation_time = time.time()
            observation = env.get_observations()
            after_observation_time = time.time()
            observation_time= after_observation_time - before_observation_time
            # print("observation time: ", observation_time)
            Fz=observation["ft"][2]
            if Fz<=-40:
                print(f"stopped rollout because downward force was to big: fz = {Fz}")
                env.move_robot_to_tcp_pose(pre_touching_pose)
                raise ValueError("stopped rollout because downward force was to big")
               

            # update & handle state machine events
            if not state.is_recording and event.start_recording:

                state.is_recording = True
                print("start recording")
                dataset_recorder.start_episode()
                # Start Phase 1
                state.phase1 = True
                state.phase2 = False
                state.phase3 = False  

            if not state.is_recording:
                # Move to start pose
                pose_idx = dataset_recorder.n_recorded_episodes % len(deterministic_poses)
                new_pose = deterministic_poses[pose_idx]
                print(f"Moving to new pose for episode {dataset_recorder.n_recorded_episodes}: {new_pose}")
                move_to_start_pose_with_negative_z_detour(new_pose)

                # Generate Random Style
                # style < 1.0: Aggressive Z (Go down first)
                # style > 1.0: Aggressive XY (Align first)

                approach_style = np.exp(np.random.uniform(np.log(0.1), np.log(10.0)))
                # approach_style = styles[style_idx]
                # Calculate Gains based on style
                k_xy = BASE_LATERAL_GAIN * np.sqrt(approach_style)
                k_z  = BASE_SPEED_GAIN / np.sqrt(approach_style)
                k_rot = BASE_ROT_GAIN * np.sqrt(approach_style)

                print(
                    f"Episode Style: {approach_style:.4f} "
                    f"(Lateral K: {k_xy:.3f}, Vert K: {k_z:.3f}, Rot K: {k_rot:.3f})"
                )
                event.start_recording=True
                continue

            elif event.quit:
                print("quit")
                state.is_stopped = True
                listener.stop()
                dataset_recorder.finish_recording()
                return

            # clear all events
            event.clear()

            # update GUI.
            vis_img = observation["wrist_image"].copy()

            # visualize state is_recording, is_paused
            if state.is_recording:
                cv2.putText(vis_img, "RECORDING", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if state.is_paused:
                cv2.putText(vis_img, "PAUSED", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(
                vis_img,
                f" # episodes: {dataset_recorder.n_recorded_episodes}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            rr.log("wrist_image", rr.Image(vis_img, rr.ColorModel.RGB))
            rr.log("wrist_image_original", rr.Image(observation["wrist_image_original"], rr.ColorModel.RGB))
            rr.log("spectogram", rr.Image(observation["spectogram_image"], rr.ColorModel.RGB))
            rr.log("spectogramgray", rr.Image(observation["spectogram_values"], rr.ColorModel.RGB))
            # rr.log("spectogrambgr", rr.Image(observation["spectogram_image"], rr.ColorModel.BGR))
            # print(f"wrist image shape: {observation['wrist_image'].shape}")
            # print(f"spectogram image shape: {observation['spectogram_image'].shape}")
            # rr.log("scene_image", rr.Image(observation["scene_image"], rr.ColorModel.RGB))
            rr.log("joints", rr.TextLog(str(observation["joints"])))
            rr.log("btn_state", rr.Scalars(float(observation["btn_state"])))
            if state.is_recording:

                # A. Update Vectors (Calculated every frame for all phases)
                current_tcp_pose = env.get_robot_pose_se3()
                vec_to_button = create_action(
                    current_tcp_pose,
                    target_position,
                )
                
                lateral_error = np.linalg.norm(vec_to_button[:2])
                z_dist_to_pre = vec_to_button[2] - BUTTON_PRE_Z_OFFSET
                z_dist_actual = vec_to_button[2] # Actual distance to button center

                target_rotation = pre_touching_pose[:3, :3]
                current_rotation = current_tcp_pose[:3, :3]
                # Relative orientation in matrix form (tool frame): R_rel = R_curr^{-1} R_target.
                R_error_tool = current_rotation.T @ target_rotation
                rotvec_error_tool = SE3Container.from_rotation_matrix_and_translation(
                    R_error_tool,
                    np.zeros(3),
                ).orientation_as_rotation_vector
                orientation_error = np.linalg.norm(rotvec_error_tool)

                # B. Phase Switching Logic
                if state.phase1 and lateral_error < LATERAL_TOLERANCE and orientation_error < ORIENTATION_TOLERANCE:
                    print("Phase 1 Complete -> Phase 2 (Descent)")
                    state.phase1 = False
                    state.phase2 = True
                elif state.phase2 and observation["btn_state"] == 0: # Button Pushed
                    print("Button Pushed -> Phase 3 (Retract)")
                    # button is pushed
                    state.phase2=False
                    state.phase3=True

                elif state.phase3 and z_dist_actual >=RETRACT_HEIGHT_OFFSET:
                    print("Retraction Complete -> Resetting")
                    state.phase1=False
                    state.phase2=False
                    state.phase3=False
                    dataset_recorder.save_episode()
                    # event.start_recording=True # trigger next loop
                    state.is_recording = False # Loop finished
                    if pose_idx == len(deterministic_poses) - 1:
                        print("All deterministic poses completed.")
                        raise ValueError("All deterministic poses completed")
                    # Prepare next episode
                    # style_idx += 1
                    # if style_idx >= len(styles):
                    #     print("All styles done for this pose. Generating new pose...")
                    #     style_idx = 0
                    #     # Generate new pose
                    #     new_pose = randomize_initial_pose(pre_touching_pose)
                    #     attempts = 0
                    #     while not env.is_tcp_pose_reachable(new_pose) and attempts < 20:
                    #         new_pose = randomize_initial_pose(pre_touching_pose)
                    #         attempts += 1
                    #     if attempts == 20:
                    #         print("no valid initial pose found, please shift the button")
                    #         raise ValueError
                # C. Execution Logic
                step_action = np.zeros(6)

                if state.phase1:
                    # print("I'm in state 1")
                    # 4. Calculate Steps
                    step_action = np.zeros(6)
                    
                    # --- XY Action ---
                    # Magnitude based on distance * gain
                    xy_step_mag = lateral_error * k_xy
                    xy_step_mag = np.clip(xy_step_mag, MIN_STEP, MAX_STEP)
                    xy_dist_vector = vec_to_button[:2] 

                    # Normalize vector and apply magnitude
                    if lateral_error > LATERAL_TOLERANCE:
                        step_action[:2] = (xy_dist_vector / lateral_error) * xy_step_mag
                    
                    # --- Z Action ---
                    z_step_mag = abs(z_dist_to_pre) * k_z
                    z_step_mag = np.clip(z_step_mag, MIN_STEP, MAX_STEP)
                    if abs(z_dist_to_pre)<=MIN_STEP:#add null action to remove jitter
                        z_step_mag = 0
                    step_action[2] = np.sign(z_dist_to_pre) * z_step_mag

                    # --- Rotation Action ---
                    if orientation_error > ORIENTATION_TOLERANCE:
                        rot_step_mag = orientation_error * k_rot
                        rot_step_mag = np.clip(rot_step_mag, MIN_ROT_STEP, MAX_ROT_STEP)
                        if orientation_error <= MIN_ROT_STEP:
                            rot_step_mag = 0
                        step_action[3:6] = (rotvec_error_tool / orientation_error) * rot_step_mag

                    # 5. Execute
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                    # Record the adaptive action
                    # if not (observation["wrist_image_original"] == observation["wrist_image"]).all():
                    #     raise ValueError
                    
                elif state.phase2:
                    # print("Phase 1 Complete (Aligned). Starting Fixed Descent...")
                    if z_dist_to_pre>0:
                        z_step_mag = abs(z_dist_to_pre) * k_z
                        z_step_mag = np.clip(z_step_mag, 0.005, MAX_STEP)
                        step_action[2] = z_step_mag
                    else:  
                        # Use MIN_STEP (0.001) constant speed
                        # step_action[2] = np.sign(vec_to_button[2]) * MIN_STEP
                        step_action[2] =  0.001
                        
                    # 3. Execute
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                elif state.phase3:
                    #here we want to go up untill about 10 cm above the button, first fast, then slowly
                    step_action[2] = -0.02 #2 cm up
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                # print(step_action)
            # wait for end of the control period
            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)
            else:
                print("cycle time exceeded control period")
    finally:
        dataset_recorder.finish_recording()


def collect_data_xyz_red_switch(  # noqa: C901
    env: BaseEnv,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
    detected_3d_button_positions = None,
    pre_touching_pose: HomogeneousMatrixType = None,
    deterministic_poses:list[HomogeneousMatrixType]=None,
):
    # Local import avoids top-level circular imports: eval_agent_delta_z imports from this module.
    from robot_imitation_glue.eval_agent_delta_z import generate_deterministic_poses

    rr.init("robot_imitation_glue", spawn=True)
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency


    # from airo_dataset_tools.data_parsers.pose import Pose
    # camera_pose_path = "camera_pose_Daniilidis.json"

    # with open(camera_pose_path, "r") as f:
    #     camera_pose = Pose.model_validate_json(f.read()).as_homogeneous_matrix()


    # def collect_rgbd_and_tcp_pose(joints: JointConfigurationType
    # ) -> Tuple[NumpyIntImageType, NumpyDepthMapType, HomogeneousMatrixType]:
    #     env.move_robot_to_joint_config(joints,0,wait=True)
    #     observation = env.get_observations()
    #     image = observation["wrist_image_original"]
    #     depth_map = env.get_depth_map()
    #     X_B_TCP = env.get_robot_pose_se3()
    #     return image, depth_map, X_B_TCP
    # # collect images
    # image_depth_X_B_TCP = [collect_rgbd_and_tcp_pose(joints) for joints in button_detector.joint_positions] 

    # # detect button
    # image_and_detections=[]
    # for img,depth,X_B_TCP in image_depth_X_B_TCP:
    #     results = button_detector.detect_button_ML(img,depth,X_B_TCP )
    #     if results:
    #         image_and_detections.append(results)
    # # calculate button position in world frame
    # detected_3d_button_positions = button_detector.get_3d_coordinates_of_pixels_with_depth(image_and_detections)
    # go to prethouch pose

    
    # ########################3
    # theta = np.arange(0,-np.pi, -np.pi/4) # 4 different angles
    # for t in theta:
    #     Rz = np.array([
    #         [np.cos(t), -np.sin(t), 0, 0],
    #         [np.sin(t),  np.cos(t), 0, 0],
    #         [0,               0,            1, 0],
    #         [0,               0,            0, 1],
    #     ])

    #     X_rotated = pre_touching_pose @ Rz
    #     env.move_robot_to_tcp_pose(X_rotated)
    #     time.sleep(5)
    # raise ValueError("done testing rotations, now randomizing initial pose")
    # ########################3
    if len(deterministic_poses) == 0:
        raise ValueError("No deterministic reachable poses found, please shift the button")

    def move_to_start_pose_with_negative_z_detour(target_pose):
        """
        Move to target start pose with a deterministic orientation policy:
        - If the target requires a positive yaw turn larger than 90 deg from current,
          first rotate -90 deg around tool Z at the current position.
        - Otherwise, move directly to target pose.
        """
        R1 = pre_touching_pose[:3, :3]
        R2 = target_pose[:3, :3]
        R_diff = R1.T @ R2
        angle = np.arccos((np.trace(R_diff) - 1) / 2)
        print("angle between R1 and R2:", np.degrees(angle), "degrees")
        if np.degrees(angle)>90:
            print("applying detour because angle is greater than 90 degrees")
            Rz_minus_90 = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
            detour_pose = target_pose.copy()
            detour_pose[:3, :3] = Rz_minus_90

            if env.is_tcp_pose_reachable(detour_pose):
                print("Applying negative-Z yaw detour before start pose")
                env.move_robot_to_tcp_pose(detour_pose)

        env.move_robot_to_tcp_pose(target_pose)

    pose_idx = dataset_recorder.n_recorded_episodes % len(deterministic_poses)
    new_pose = deterministic_poses[pose_idx]
    move_to_start_pose_with_negative_z_detour(new_pose)

    # observation = env.get_observations()
    # image = observation["wrist_image_original"]
    # BUTTON_Z = 0.036 
    # button_position = button_detector.get_button_position_world_height_known(image,BUTTON_Z,pre_touching_pose)

    # --- Configuration for Adaptive Speed ---
    target_position = detected_3d_button_positions 
    # Simulation Constants adapted for Robot Safety
    BUTTON_PRE_Z_OFFSET = 0.015 # 1cm above button
    LATERAL_TOLERANCE   = 0.002 # 1mm (When we consider X/Y aligned)
    MIN_STEP = 0.002  # 1mm min step
    MAX_STEP = 0.03   # 1cm max step (Robot safety cap)
    #best values from sim
    # MIN_STEP = 0.00001
    # MAX_STEP = 0.04
    BASE_SPEED_GAIN = 0.1
    BASE_LATERAL_GAIN = 0.1
    BASE_ROT_GAIN = 0.1
    RETRACT_HEIGHT_OFFSET = 0.10 # 10cm above button
    GRIPPER_EDGE_OFFSET_TOOL = np.array([0.0, -0.0152, 0.0])
    ORIENTATION_TOLERANCE = np.deg2rad(5.0)
    MIN_ROT_STEP = np.deg2rad(0.5)
    MAX_ROT_STEP = np.deg2rad(3.0)
    # ----------------------------------------

    # styles = np.logspace(np.log10(0.1), np.log10(10.0), 10)
    # style_idx=0

    k_xy = BASE_LATERAL_GAIN 
    k_z  = BASE_SPEED_GAIN
    k_rot = BASE_ROT_GAIN

    event.start_recording=True
    try:

        while not state.is_stopped:
            cycle_end_time = time.time() + control_period

            before_observation_time = time.time()
            observation = env.get_observations()
            after_observation_time = time.time()
            observation_time= after_observation_time - before_observation_time
            # print("observation time: ", observation_time)
            Fz=observation["ft"][2]
            if Fz<=-40:
                print(f"stopped rollout because downward force was to big: fz = {Fz}")
                env.move_robot_to_tcp_pose(pre_touching_pose)
                raise ValueError("stopped rollout because downward force was to big")
               

            # update & handle state machine events
            if not state.is_recording and event.start_recording:

                state.is_recording = True
                print("start recording")
                dataset_recorder.start_episode()
                # Start Phase 1
                state.phase1 = True
                state.phase2 = False
                state.phase3 = False  

            if not state.is_recording:
                # Move to start pose
                pose_idx = dataset_recorder.n_recorded_episodes % len(deterministic_poses)
                new_pose = deterministic_poses[pose_idx]
                print(f"Moving to new pose for episode {dataset_recorder.n_recorded_episodes}: {new_pose}")
                move_to_start_pose_with_negative_z_detour(new_pose)

                # Generate Random Style
                # style < 1.0: Aggressive Z (Go down first)
                # style > 1.0: Aggressive XY (Align first)

                approach_style = np.exp(np.random.uniform(np.log(0.1), np.log(10.0)))
                # approach_style = styles[style_idx]
                # Calculate Gains based on style
                k_xy = BASE_LATERAL_GAIN * np.sqrt(approach_style)
                k_z  = BASE_SPEED_GAIN / np.sqrt(approach_style)
                k_rot = BASE_ROT_GAIN * np.sqrt(approach_style)

                print(
                    f"Episode Style: {approach_style:.4f} "
                    f"(Lateral K: {k_xy:.3f}, Vert K: {k_z:.3f}, Rot K: {k_rot:.3f})"
                )
                event.start_recording=True
                continue

            elif event.quit:
                print("quit")
                state.is_stopped = True
                listener.stop()
                dataset_recorder.finish_recording()
                return

            # clear all events
            event.clear()

            # update GUI.
            vis_img = observation["wrist_image"].copy()

            # visualize state is_recording, is_paused
            if state.is_recording:
                cv2.putText(vis_img, "RECORDING", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if state.is_paused:
                cv2.putText(vis_img, "PAUSED", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(
                vis_img,
                f" # episodes: {dataset_recorder.n_recorded_episodes}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            rr.log("wrist_image", rr.Image(vis_img, rr.ColorModel.RGB))
            rr.log("wrist_image_original", rr.Image(observation["wrist_image_original"], rr.ColorModel.RGB))
            rr.log("spectogram", rr.Image(observation["spectogram_image"], rr.ColorModel.RGB))
            rr.log("spectogramgray", rr.Image(observation["spectogram_values"], rr.ColorModel.RGB))
            # rr.log("spectogrambgr", rr.Image(observation["spectogram_image"], rr.ColorModel.BGR))
            # print(f"wrist image shape: {observation['wrist_image'].shape}")
            # print(f"spectogram image shape: {observation['spectogram_image'].shape}")
            # rr.log("scene_image", rr.Image(observation["scene_image"], rr.ColorModel.RGB))
            rr.log("joints", rr.TextLog(str(observation["joints"])))
            rr.log("btn_state", rr.Scalars(float(observation["btn_state"])))
            if state.is_recording:

                # A. Update Vectors (Calculated every frame for all phases)
                current_tcp_pose = env.get_robot_pose_se3()
                vec_to_button = create_action_from_tool_offset(
                    current_tcp_pose,
                    target_position,
                    GRIPPER_EDGE_OFFSET_TOOL,
                )
                
                lateral_error = np.linalg.norm(vec_to_button[:2])
                z_dist_to_pre = vec_to_button[2] - BUTTON_PRE_Z_OFFSET
                z_dist_actual = vec_to_button[2] # Actual distance to button center

                target_rotation = pre_touching_pose[:3, :3]
                current_rotation = current_tcp_pose[:3, :3]
                # Relative orientation in matrix form (tool frame): R_rel = R_curr^{-1} R_target.
                R_error_tool = current_rotation.T @ target_rotation
                rotvec_error_tool = SE3Container.from_rotation_matrix_and_translation(
                    R_error_tool,
                    np.zeros(3),
                ).orientation_as_rotation_vector
                orientation_error = np.linalg.norm(rotvec_error_tool)

                # B. Phase Switching Logic
                if state.phase1 and lateral_error < LATERAL_TOLERANCE and orientation_error < ORIENTATION_TOLERANCE:
                    print("Phase 1 Complete -> Phase 2 (Descent)")
                    state.phase1 = False
                    state.phase2 = True
                elif state.phase2 and observation["btn_state"] == 0: # Button Pushed
                    print("Button Pushed -> Phase 3 (Retract)")
                    # button is pushed
                    state.phase2=False
                    state.phase3=True

                elif state.phase3 and z_dist_actual >=RETRACT_HEIGHT_OFFSET:
                    print("Retraction Complete -> Resetting")
                    state.phase1=False
                    state.phase2=False
                    state.phase3=False
                    dataset_recorder.save_episode()
                    # event.start_recording=True # trigger next loop
                    state.is_recording = False # Loop finished
                    if pose_idx == len(deterministic_poses) - 1:
                        print("All deterministic poses completed.")
                        raise ValueError("All deterministic poses completed")
                    # Prepare next episode
                    # style_idx += 1
                    # if style_idx >= len(styles):
                    #     print("All styles done for this pose. Generating new pose...")
                    #     style_idx = 0
                    #     # Generate new pose
                    #     new_pose = randomize_initial_pose(pre_touching_pose)
                    #     attempts = 0
                    #     while not env.is_tcp_pose_reachable(new_pose) and attempts < 20:
                    #         new_pose = randomize_initial_pose(pre_touching_pose)
                    #         attempts += 1
                    #     if attempts == 20:
                    #         print("no valid initial pose found, please shift the button")
                    #         raise ValueError
                # C. Execution Logic
                step_action = np.zeros(6)

                if state.phase1:
                    # print("I'm in state 1")
                    # 4. Calculate Steps
                    step_action = np.zeros(6)
                    
                    # --- XY Action ---
                    # Magnitude based on distance * gain
                    xy_step_mag = lateral_error * k_xy
                    xy_step_mag = np.clip(xy_step_mag, MIN_STEP, MAX_STEP)
                    xy_dist_vector = vec_to_button[:2] 

                    # Normalize vector and apply magnitude
                    if lateral_error > LATERAL_TOLERANCE:
                        step_action[:2] = (xy_dist_vector / lateral_error) * xy_step_mag
                    
                    # --- Z Action ---
                    z_step_mag = abs(z_dist_to_pre) * k_z
                    z_step_mag = np.clip(z_step_mag, MIN_STEP, MAX_STEP)
                    if abs(z_dist_to_pre)<=MIN_STEP:#add null action to remove jitter
                        z_step_mag = 0
                    step_action[2] = np.sign(z_dist_to_pre) * z_step_mag

                    # --- Rotation Action ---
                    if orientation_error > ORIENTATION_TOLERANCE:
                        rot_step_mag = orientation_error * k_rot
                        rot_step_mag = np.clip(rot_step_mag, MIN_ROT_STEP, MAX_ROT_STEP)
                        if orientation_error <= MIN_ROT_STEP:
                            rot_step_mag = 0
                        step_action[3:6] = (rotvec_error_tool / orientation_error) * rot_step_mag

                    # 5. Execute
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                    # Record the adaptive action
                    # if not (observation["wrist_image_original"] == observation["wrist_image"]).all():
                    #     raise ValueError
                    
                elif state.phase2:
                    # print("Phase 1 Complete (Aligned). Starting Fixed Descent...")
                    if z_dist_to_pre>0:
                        z_step_mag = abs(z_dist_to_pre) * k_z
                        z_step_mag = np.clip(z_step_mag, 0.005, MAX_STEP)
                        step_action[2] = z_step_mag
                    else:  
                        # Use MIN_STEP (0.001) constant speed
                        # step_action[2] = np.sign(vec_to_button[2]) * MIN_STEP
                        step_action[2] =  0.001
                        
                    # 3. Execute
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                elif state.phase3:
                    #here we want to go up untill about 10 cm above the button, first fast, then slowly
                    step_action[2] = -0.02 #2 cm up
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                # print(step_action)
            # wait for end of the control period
            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)
            else:
                print("cycle time exceeded control period")
    finally:
        dataset_recorder.finish_recording()

def collect_data_xyz(  # noqa: C901
    env: BaseEnv,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
    detected_3d_button_positions = None,
    pre_touching_pose: HomogeneousMatrixType = None,
    deterministic_poses:list[HomogeneousMatrixType]=None,
):
    # Local import avoids top-level circular imports: eval_agent_delta_z imports from this module.
    # from robot_imitation_glue.eval_agent_delta_z import generate_deterministic_poses

    rr.init("robot_imitation_glue")
    rr.spawn(memory_limit="10GB")
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency


    # from airo_dataset_tools.data_parsers.pose import Pose
    # camera_pose_path = "camera_pose_Daniilidis.json"

    # with open(camera_pose_path, "r") as f:
    #     camera_pose = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

    # button_detector = ButtonDetector(env.get_camera_intrinsics(),None)

    # def collect_rgbd_and_tcp_pose(joints: JointConfigurationType
    # ) -> Tuple[NumpyIntImageType, NumpyDepthMapType, HomogeneousMatrixType]:
    #     env.move_robot_to_joint_config(joints,0,wait=True)
    #     observation = env.get_observations()
    #     image = observation["wrist_image_original"]
    #     depth_map = env.get_depth_map()
    #     X_B_TCP = env.get_robot_pose_se3()
    #     return image, depth_map, X_B_TCP
    # # collect images
    # image_depth_X_B_TCP = [collect_rgbd_and_tcp_pose(joints) for joints in button_detector.joint_positions] 

    # # detect button
    # image_and_detections=[]
    # for img,depth,X_B_TCP in image_depth_X_B_TCP:
    #     results = button_detector.detect_button_ML(img,depth,X_B_TCP )
    #     if results:
    #         image_and_detections.append(results)
    # # calculate button position in world frame
    # detected_3d_button_positions = button_detector.get_3d_coordinates_of_pixels_with_depth(image_and_detections)
    # go to prethouch pose
    # env.robot.rtde_control.teachMode()
    # input("go above button position")
    # detected_3d_button_positions = env.get_robot_pose_se3()[:3,3]
    # # detected_3d_button_positions = [-0.49433436 ,-0.0376252   ,0.06231323]#red round button small
    # # #env.get_robot_pose_se3()[:3,3]
    # env.robot.rtde_control.endTeachMode()
    # print(f"detected button position = {detected_3d_button_positions}")
    # pre_touching_pose = button_detector.get_pretouch_position(detected_3d_button_positions)
    # ########################3
    # theta = np.arange(0,-np.pi, -np.pi/4) # 4 different angles
    # for t in theta:
    #     Rz = np.array([
    #         [np.cos(t), -np.sin(t), 0, 0],
    #         [np.sin(t),  np.cos(t), 0, 0],
    #         [0,               0,            1, 0],
    #         [0,               0,            0, 1],
    #     ])
    #     X_rotated = pre_touching_pose @ Rz
    #     env.move_robot_to_tcp_pose(X_rotated)
    #     time.sleep(5)
    # raise ValueError("done testing rotations, now randomizing initial pose")
    # ########################3
    # deterministic_poses = generate_deterministic_poses(pre_touching_pose, env, count=100,seed=18)
    # deterministic_poses = generate_deterministic_poses(pre_touching_pose, env, count=25,seed=19)
    if len(deterministic_poses) == 0:
        raise ValueError("No deterministic reachable poses found, please shift the button")

    pose_idx = dataset_recorder.n_recorded_episodes % len(deterministic_poses)
    new_pose = deterministic_poses[pose_idx]
    env.move_robot_to_tcp_pose(new_pose)

    # observation = env.get_observations()
    # image = observation["wrist_image_original"]
    # BUTTON_Z = 0.036 
    # button_position = button_detector.get_button_position_world_height_known(image,BUTTON_Z,pre_touching_pose)

    # --- Configuration for Adaptive Speed ---
    target_position = detected_3d_button_positions 
    # Simulation Constants adapted for Robot Safety
    BUTTON_PRE_Z_OFFSET = 0.015 # 1cm above button
    LATERAL_TOLERANCE   = 0.002 # 1mm (When we consider X/Y aligned)
    MIN_STEP = 0.002  # 1mm min step
    MAX_STEP = 0.03   # 1cm max step (Robot safety cap)
    #best values from sim
    # MIN_STEP = 0.00001
    # MAX_STEP = 0.04
    BASE_SPEED_GAIN = 0.1
    BASE_LATERAL_GAIN = 0.1
    RETRACT_HEIGHT_OFFSET = 0.10 # 10cm above button
    # ----------------------------------------
    extra_steps=1
    # styles = np.logspace(np.log10(0.1), np.log10(10.0), 10)
    # style_idx=0

    k_xy = BASE_LATERAL_GAIN 
    k_z  = BASE_SPEED_GAIN

    event.start_recording=True
    try:

        while not state.is_stopped:
            cycle_end_time = time.time() + control_period

            before_observation_time = time.time()
            observation = env.get_observations()
            after_observation_time = time.time()
            observation_time= after_observation_time - before_observation_time
            # print("observation time: ", observation_time)
            Fz=observation["ft"][2]
            # if Fz<=-40:
            #     print(f"stopped rollout because downward force was to big: fz = {Fz}")
            #     env.move_robot_to_tcp_pose(pre_touching_pose)
            #     raise ValueError("stopped rollout because downward force was to big")
               

            # update & handle state machine events
            if not state.is_recording and event.start_recording:

                state.is_recording = True
                print("start recording")
                dataset_recorder.start_episode()
                # Start Phase 1
                state.phase1 = True
                state.phase2 = False
                state.phase3 = False  

            if not state.is_recording:
                #reset button env
                # current_tcp_pose = env.get_robot_pose_se3()
                # pre_reset_pose = current_tcp_pose.copy()
                # pre_reset_pose[2,3] = detected_3d_button_positions[2] 
                # reset_pose = pre_reset_pose.copy()
                # reset_pose[2,3]-= 0.007 
                # env.move_robot_to_tcp_pose(pre_reset_pose)
                # env.move_robot_to_tcp_pose(reset_pose)
                # env.move_robot_to_tcp_pose(pre_reset_pose)

                # Move to start pose
                pose_idx = dataset_recorder.n_recorded_episodes % len(deterministic_poses)
                new_pose = deterministic_poses[pose_idx]
                print(f"Moving to new pose for episode {dataset_recorder.n_recorded_episodes}: {new_pose}")
                env.move_robot_to_tcp_pose(new_pose)

                # Generate Random Style
                # style < 1.0: Aggressive Z (Go down first)
                # style > 1.0: Aggressive XY (Align first)

                approach_style = np.exp(np.random.uniform(np.log(0.1), np.log(10.0)))
                # approach_style = styles[style_idx]
                # Calculate Gains based on style
                k_xy = BASE_LATERAL_GAIN * np.sqrt(approach_style)
                k_z  = BASE_SPEED_GAIN / np.sqrt(approach_style)

                print(f"Episode Style: {approach_style:.4f} (Lateral K: {k_xy:.3f}, Vert K: {k_z:.3f})")
                event.start_recording=True
                continue

            elif event.quit:
                print("quit")
                state.is_stopped = True
                listener.stop()
                dataset_recorder.finish_recording()
                return

            # clear all events
            event.clear()

            # update GUI.
            vis_img = observation["wrist_image"].copy()

            # visualize state is_recording, is_paused
            if state.is_recording:
                cv2.putText(vis_img, "RECORDING", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if state.is_paused:
                cv2.putText(vis_img, "PAUSED", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(
                vis_img,
                f" # episodes: {dataset_recorder.n_recorded_episodes}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            rr.log("wrist_image", rr.Image(vis_img, rr.ColorModel.RGB))
            rr.log("wrist_image_original", rr.Image(observation["wrist_image_original"], rr.ColorModel.RGB))
            rr.log("spectogram", rr.Image(observation["spectogram_image"], rr.ColorModel.RGB))
            rr.log("spectogramgray", rr.Image(observation["spectogram_values"], rr.ColorModel.RGB))
            # rr.log("spectogrambgr", rr.Image(observation["spectogram_image"], rr.ColorModel.BGR))
            # print(f"wrist image shape: {observation['wrist_image'].shape}")
            # print(f"spectogram image shape: {observation['spectogram_image'].shape}")
            # rr.log("scene_image", rr.Image(observation["scene_image"], rr.ColorModel.RGB))
            rr.log("joints", rr.TextLog(str(observation["joints"])))
            rr.log("btn_state", rr.Scalars(float(observation["btn_state"])))
            rr.log("fz", rr.Scalars(float(Fz)))

            if state.is_recording:

                # A. Update Vectors (Calculated every frame for all phases)
                current_tcp_pose = env.get_robot_pose_se3()
                vec_to_button = create_action(current_tcp_pose, target_position)
                
                lateral_error = np.linalg.norm(vec_to_button[:2])
                z_dist_to_pre = vec_to_button[2] - BUTTON_PRE_Z_OFFSET
                z_dist_actual = vec_to_button[2] # Actual distance to button center
                # B. Phase Switching Logic
                if state.phase1 and lateral_error < LATERAL_TOLERANCE:
                    print("Phase 1 Complete -> Phase 2 (Descent)")
                    state.phase1 = False
                    state.phase2 = True
                elif state.phase2 and observation["btn_state"] == 0: # Button Pushed
                    print("Button Pushed -> Phase 3 (Retract)")
                    # button is pushed
                    extra_steps-=1
                    if(extra_steps==0):
                        state.phase2=False
                        state.phase3=True
                        extra_steps=1

                elif state.phase3 and z_dist_actual >=RETRACT_HEIGHT_OFFSET:
                    print("Retraction Complete -> Resetting")
                    state.phase1=False
                    state.phase2=False
                    state.phase3=False
                    dataset_recorder.save_episode()
                    # event.start_recording=True # trigger next loop
                    state.is_recording = False # Loop finished
                    if pose_idx == len(deterministic_poses) - 1:
                        print("All deterministic poses completed.")
                        raise ValueError("All deterministic poses completed")
                    # Prepare next episode
                    # style_idx += 1
                    # if style_idx >= len(styles):
                    #     print("All styles done for this pose. Generating new pose...")
                    #     style_idx = 0
                    #     # Generate new pose
                    #     new_pose = randomize_initial_pose(pre_touching_pose)
                    #     attempts = 0
                    #     while not env.is_tcp_pose_reachable(new_pose) and attempts < 20:
                    #         new_pose = randomize_initial_pose(pre_touching_pose)
                    #         attempts += 1
                    #     if attempts == 20:
                    #         print("no valid initial pose found, please shift the button")
                    #         raise ValueError
                # C. Execution Logic
                step_action = np.zeros(6)

                if state.phase1:
                    # print("I'm in state 1")
                    # 4. Calculate Steps
                    step_action = np.zeros(6)
                    
                    # --- XY Action ---
                    # Magnitude based on distance * gain
                    xy_step_mag = lateral_error * k_xy
                    xy_step_mag = np.clip(xy_step_mag, MIN_STEP, MAX_STEP)
                    xy_dist_vector = vec_to_button[:2] 

                    # Normalize vector and apply magnitude
                    if lateral_error > 0:
                        step_action[:2] = (xy_dist_vector / lateral_error) * xy_step_mag
                    
                    # --- Z Action ---
                    z_step_mag = abs(z_dist_to_pre) * k_z
                    z_step_mag = np.clip(z_step_mag, MIN_STEP, MAX_STEP)
                    if abs(z_dist_to_pre)<=MIN_STEP:#add null action to remove jitter
                        z_step_mag = 0
                    step_action[2] = np.sign(z_dist_to_pre) * z_step_mag

                    # 5. Execute
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                    # Record the adaptive action
                    # if not (observation["wrist_image_original"] == observation["wrist_image"]).all():
                    #     raise ValueError
                    
                elif state.phase2:
                    # print("Phase 1 Complete (Aligned). Starting Fixed Descent...")
                    if z_dist_to_pre>0:
                        z_step_mag = abs(z_dist_to_pre) * k_z
                        z_step_mag = np.clip(z_step_mag, 0.005, MAX_STEP)
                        step_action[2] = z_step_mag
                    else:  
                        # Use MIN_STEP (0.001) constant speed
                        # step_action[2] = np.sign(vec_to_button[2]) * MIN_STEP
                        # step_action[2] =  0.015 for big red button
                        step_action[2] =  0.001
                        
                    # 3. Execute
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                elif state.phase3:
                    #here we want to go up untill about 10 cm above the button, first fast, then slowly
                    step_action[2] = -0.02 #2 cm up
                    policy_action = step_action_to_policy_action_6d(step_action)
                    next_pose = policy_action_to_tcp_pose(current_tcp_pose, policy_action)
                    dataset_recorder.record_step(observation, policy_action)
                    env.act_tcp(next_pose, time.time() + control_period)

                # print(step_action)
                rr.log("step_Action", rr.Scalars(step_action[2]))

            # wait for end of the control period
            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)
            else:
                print("cycle time exceeded control period")
    finally:
        dataset_recorder.finish_recording()

def teleoperate(  # noqa: C901
    env: BaseEnv,
    teleop_agent: BaseAgent
):
    assert env.ACTION_SPEC == teleop_agent.ACTION_SPEC

    while True:
        frequency=10
        control_period = 1/frequency
        cycle_end_time = time.time() + control_period

        # print("observation time: ", observation_time)

        action = teleop_agent.get_action({})
        logger.info(f"Action: {action}")

        gripper_target = 0
        env.act(
            robot_joints=action[0:6],
            gripper_pose=gripper_target,
            timestamp=time.time() + control_period,
        )

        # wait for end of the control period
        if cycle_end_time > time.time():
            precise_wait(cycle_end_time)
        else:
            print("cycle time exceeded control period")


if __name__ == "__main__":
    # create dummy env, agent and recorder to test flow.
    from robot_imitation_glue.mock import MockAgent, MockEnv

    env = MockEnv()
    agent = MockAgent()

    dataset_name = "test_dataset"
