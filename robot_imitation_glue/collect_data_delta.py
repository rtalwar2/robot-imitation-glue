import time

# create type for callable that takes obs and returns action
from typing import Callable , Any, List, Dict, Optional, Union, Tuple


from airo_typing import HomogeneousMatrixType, JointConfigurationType, NumpyDepthMapType, NumpyIntImageType
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
    theta = np.random.uniform(-np.pi/2, np.pi/2)

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

def action_to_tcp_pose(tcp_pose,action):
    final_pose = tcp_pose.copy()
    action_in_base_frame = tcp_pose[:3,:3] @ action
    final_pose[:3,3]+=action_in_base_frame
    return final_pose

def collect_data_xyz(  # noqa: C901
    env: BaseEnv,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
):
    rr.init("robot_imitation_glue", spawn=True)
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency


    from airo_dataset_tools.data_parsers.pose import Pose
    camera_pose_path = "camera_pose_Daniilidis.json"

    with open(camera_pose_path, "r") as f:
        camera_pose = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

    button_detector = ButtonDetector(env.get_camera_intrinsics(),camera_pose)

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
    env.robot.rtde_control.teachMode()
    input("go above button position")
    detected_3d_button_positions = env.get_robot_pose_se3()[:3,3]
    # detected_3d_button_positions = [ 0.0675134 , -0.35472895  ,0.03802799]
    #[-0.02903111 ,-0.40018098 , 0.03766195]
    # detected_3d_button_positions =  [ 0.06598392 ,-0.35092692 , 0.03709084]
    # #env.get_robot_pose_se3()[:3,3]
    env.robot.rtde_control.endTeachMode()
    print(f"detected button position = {detected_3d_button_positions}")
    pre_touching_pose = button_detector.get_pretouch_position(detected_3d_button_positions)
    new_pose = randomize_initial_pose(pre_touching_pose)
    num_attempts = 0
    while(not env.is_tcp_pose_reachable(new_pose) and num_attempts < 20):
        num_attempts += 1
        new_pose = randomize_initial_pose(pre_touching_pose)
    if num_attempts == 20:
        print("no valid initial pose found, please shift the button")
        raise ValueError

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

    # styles = np.logspace(np.log10(0.1), np.log10(10.0), 10)
    # style_idx=0

    k_xy = BASE_LATERAL_GAIN 
    k_z  = BASE_SPEED_GAIN

    event.start_recording=True

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
            # Start Phase 1
            state.phase1 = True
            state.phase2 = False
            state.phase3 = False  

        if not state.is_recording:
            # Move to start pose
            input("are you ready for recording?")
            # print(f"Moving to start pose for Style {style_idx+1}/{len(styles)}")
            new_pose = randomize_initial_pose(pre_touching_pose)
            attempts = 0
            while not env.is_tcp_pose_reachable(new_pose) and attempts < 20:
                new_pose = randomize_initial_pose(pre_touching_pose)
                attempts += 1
            if attempts == 20:
                print("no valid initial pose found, please shift the button")
                raise ValueError
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
        rr.log("btn_state", rr.Scalar(float(observation["btn_state"])))
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
            step_action = np.zeros(3)

            if state.phase1:
                # print("I'm in state 1")
                # 4. Calculate Steps
                step_action = np.zeros(3)
                
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
                next_pose = action_to_tcp_pose(current_tcp_pose, step_action)
                dataset_recorder.record_step(observation, step_action)
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
                next_pose = action_to_tcp_pose(current_tcp_pose, step_action)
                dataset_recorder.record_step(observation, step_action)
                env.act_tcp(next_pose, time.time() + control_period)

            elif state.phase3:
                #here we want to go up untill about 10 cm above the button, first fast, then slowly
                step_action[2] = -0.02 #2 cm up
                next_pose = action_to_tcp_pose(current_tcp_pose, step_action)
                dataset_recorder.record_step(observation, step_action)
                env.act_tcp(next_pose, time.time() + control_period)

            # print(step_action)
        # wait for end of the control period
        if cycle_end_time > time.time():
            precise_wait(cycle_end_time)
        else:
            print("cycle time exceeded control period")

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
