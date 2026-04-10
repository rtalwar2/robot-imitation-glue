from airo_spatial_algebra import SE3Container

import cv2
import numpy as np
from PIL import Image

from robot_imitation_glue.button_detector.CVButtonDetector import CVButtonDetector
from robot_imitation_glue.button_detector.MLButtonDetector import MLButtonDetector


class ButtonDetector:

    def __init__(self,intrinsics,X_TCP_C,detector_id="/home/rtalwar/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb",        segmenter_id = "/home/rtalwar/.cache/huggingface/hub/models--facebook--sam-vit-base/snapshots/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
 ):


        self.joint_positions=[]
        self.joint_positions.append([1.80086112,-1.71617474,0.92627412,-1.09584095,-1.8836549,-2.63405687])
        self.joint_positions.append([0.99173224,-1.68089165,0.62467462,-1.04109986,-1.41380197,-3.40248639])
        self.joint_positions.append([0.32339218,-1.44230221,0.62475044,-0.78069885,-1.21161396,-4.29869467])
        self.joint_positions.append([0.61693847,-1.40380482,0.6125663,-1.10800783,-1.3066085,-3.35280353])
        self.ML_button_detector = MLButtonDetector(detector_id,segmenter_id)
        self.intrinsics=intrinsics
        self.X_TCP_C = X_TCP_C
        self.cv_button_detector = CVButtonDetector()

    def detect_button_ML(self,img,depth,X_B_TCP):
        image_array, detections  = self.ML_button_detector.grounded_segmentation(
            image=Image.fromarray(img),
            polygon_refinement=True
        )
        det_filtered = self.ML_button_detector.filter_detections_by_shape_and_color(img, detections)
        if det_filtered:
            return (image_array, det_filtered,depth,X_B_TCP )
        return None

    def get_3d_coordinates_of_pixels_with_depth(self,image_and_detections):
        all_points_B = []   # list of 3D points in camera frame
        for image, det, depth, X_B_TCP in image_and_detections:
            for d in det:
                # pixel center
                x_center = d.box.xmin + (d.box.xmax - d.box.xmin) / 2
                y_center = d.box.ymin + (d.box.ymax - d.box.ymin) / 2

                # depth at pixel
                Z = depth[int(y_center), int(x_center)]

                # build camera ray vector (u, v, 1)
                pixel_h = np.array([x_center, y_center, 1.0])
                print(f" pixel_h {pixel_h}")
                print(f"intrinsics {self.intrinsics}")
                print(f"Z {Z}")
                # backproject into camera coordinates
                X_c = Z * (np.linalg.inv(self.intrinsics) @ pixel_h)

                # convert to homogeneous for transformation
                X_c_h = np.hstack([X_c, 1.0])

                # world (robot base) frame point
                X_B_C = X_B_TCP @ self.X_TCP_C
                X_B_h= X_B_C @ X_c_h
                if X_B_h[2]>0 and X_B_h[2]<0.5:
                    print("3D point in camera frame:", X_c)
                    print("3D point in base frame:", X_B_h[:3])
                    all_points_B.append( X_B_h[:3])
        return all_points_B

    def get_pretouch_position(self,detected_3d_button_positions):
        # p_B_TCP_touch = np.mean(np.vstack(detected_3d_button_positions), axis=0)   # the position where the TCP will touch your 3D point
        p_B_TCP_touch = detected_3d_button_positions   # the position where the TCP will touch your 3D point
        R_B_TCP_touch_X = np.array([1,0,0])  # rotation of TCP around X-axis  
        R_B_TCP_touch_Y = np.array([0,-1,0])  # rotation of TCP around Y-axis  
        R_B_TCP_touch_Z = np.array([0,0,-1])  # rotation of TCP around Z-axis  

        X_B_TCP_touch_se3 = SE3Container.from_orthogonal_base_vectors_and_translation(
            R_B_TCP_touch_X, R_B_TCP_touch_Y, R_B_TCP_touch_Z, p_B_TCP_touch
        )
        X_B_TCP_touch = X_B_TCP_touch_se3.homogeneous_matrix

        X_B_TCP_touch[:3, 3] = X_B_TCP_touch[:3, 3] + np.array([0.0, 0.0, 0.1]) # add 10 cm to the z-axis to avoid colliding
        return X_B_TCP_touch


    def get_button_position_world_height_known(self,image,BUTTON_Z,X_B_TCP):
        #we want to get button world
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # We use your previous function (ensure it is defined)
        output = self.cv_button_detector.detect_red_round_button(image_bgr)
        if output[0] == None:
            return None

        center, (error_u, error_v), pixel_dist, mask = output

        K_inv = np.linalg.inv(self.intrinsics)
        pixel_h = np.array([center[0], center[1], 1.0])

        # ray direction in camera coords
        ray_cam = K_inv @ pixel_h   # 3-vector direction (not normalized)

        X_B_C = X_B_TCP @ self.X_TCP_C
        # X_B_C = X_TCP_C
        # rotation and translation from camera -> world
        R = X_B_C[:3, :3]           # rotation
        t = X_B_C[:3, 3]            # camera origin in world coords


        # direction in world coords
        d_world = R @ ray_cam       # 3-vector

        # check d_world[2] (z component) to avoid division by zero
        dz = d_world[2]
        if abs(dz) < 1e-8:
            raise RuntimeError("Ray is parallel to plane (dz ≈ 0), cannot intersect plane z=BUTTON_Z")

        # solve for s: t_z + s * d_z = BUTTON_Z  => s = (BUTTON_Z - t_z) / d_z
        s = (BUTTON_Z - t[2]) / dz
        # optionally check s>0 (in front of camera)
        if s <= 0:
            raise RuntimeError("Intersection behind camera (s <= 0). Check BUTTON_Z or extrinsics.")

        point_world = t + s * d_world   # 3-vector world coordinates
        return point_world


    # def rotate_and_collect_button_positions(num_angels):
    #     button_positions=[]
    #     for i in range(num_angels):
    #         # Generate a random angle between 0 and 2π
    #         theta = -np.pi/2+ np.pi/num_angels*i
    #         # theta=-1.2144959500188266
    #         print(theta)
    #         # theta = 0
    #         # Rotation about Z-axis
    #         Rz = np.array([
    #             [np.cos(theta), -np.sin(theta), 0, 0],
    #             [np.sin(theta),  np.cos(theta), 0, 0],
    #             [0,               0,            1, 0],
    #             [0,               0,            0, 1]
    #         ])

    #         # Apply rotation
    #         X_rotated =X_B_TCP_touch @Rz 
    #         robot.move_to_tcp_pose(X_rotated).wait()
    #         image = camera.get_rgb_image_as_int()
    #         BUTTON_Z = 0.036
    #         button_position = get_button_position_world_height_known(image,BUTTON_Z,intrinsics)
    #         print(type(button_position))
    #         if isinstance(button_position,np.ndarray):
    #             action= create_action(robot.get_tcp_pose(),button_position)
    #             action
    #             button_positions.append(button_position)
    #     return button_positions