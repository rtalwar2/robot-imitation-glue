"""This file implements a multiprocess pub-sub for an airo-mono RGB camera.

This requires you to install the airo-camera-toolkit, which you can do by following the instructions here:
https://github.com/airo-ugent/airo-mono
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from airo_camera_toolkit.interfaces import RGBCamera , DepthCamera , RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIDL
from airo_ipc.cyclone_shm.patterns.ddsreader import DDSReader
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_ipc.framework.framework import IpcKind
from airo_ipc.framework.node import Node
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from loguru import logger


def initialize_ipc() -> None:
    """Backward-compatible no-op.

    Newer versions of ``airo-ipc`` initialize resources lazily in readers/writers.
    Existing call sites in this repository still invoke ``initialize_ipc()``.
    """
    return


@dataclass
class ResolutionIdl(IdlStruct):
    """We will send the resolution of the webcam over DDS: we need to define an IDL struct for this."""

    width: int
    height: int


@dataclass
class RGBFrame(BaseIDL):
    """We will send the RGB frames over shared memory: we need to derive from BaseIDL."""

    timestamp: np.ndarray
    rgb: np.ndarray
    intrinsics: np.ndarray

    @staticmethod
    def with_resolution(width: int, height: int):
        """We may not know the resolution of the webcam when we create the frame, so we need a factory method."""
        return RGBFrame(
            rgb=np.zeros((height, width, 3), dtype=np.uint8), intrinsics=np.zeros((3, 3)), timestamp=np.zeros((1,))
        )


@dataclass
class DepthFrame(BaseIDL):
    """Depth map + timestamp + intrinsics"""

    timestamp: np.ndarray          # shape (1,)
    depth: np.ndarray              # 2D float32 depth map
    # depth_image: np.ndarray        # 2D uint8 visualization
    intrinsics: np.ndarray

    @staticmethod
    def with_resolution(width: int, height: int):
        return DepthFrame(
            depth=np.zeros((height, width), dtype=np.float32),
            # depth_image=np.zeros((height, width), dtype=np.uint8),
            intrinsics=np.zeros((3, 3)),
            timestamp=np.zeros((1,))
        )

class RGBCameraPublisher(Node):
    def __init__(self, camera_creation_fn, rgb_topic_name, depth_topic_name,
                 resolution_topic_name, update_frequency, verbose=False):

        self._camera_creation_fn = camera_creation_fn
        self._camera = None
        self._rgb_topic_name = rgb_topic_name
        self._depth_topic_name = depth_topic_name
        self._resolution_topic_name = resolution_topic_name

        super().__init__(update_frequency, verbose)

    def _setup(self):
        logger.info("Opening camera.")
        self._camera = self._camera_creation_fn()
        assert isinstance(self._camera, RGBDCamera), "Camera must supply RGB & Depth"

        logger.info("Getting resolution.")
        width, height = self._camera.resolution

        logger.info("Registering publishers.")
        self._register_publisher(self._resolution_topic_name, ResolutionIdl, IpcKind.DDS)
        self._register_publisher(
            self._rgb_topic_name,
            RGBFrame.with_resolution(width, height),
            IpcKind.SHARED_MEMORY
        )

        self._register_publisher(
            self._depth_topic_name,
            DepthFrame.with_resolution(width, height),
            IpcKind.SHARED_MEMORY
        )

    def _step(self):
        rgb = self._camera.get_rgb_image_as_int()
        depth = self._camera.get_depth_map()
        # depth_img = self._camera.get_depth_image()
        intrinsics = self._camera.intrinsics_matrix()
        now = time.time()

        # resolution
        self._publish(
            self._resolution_topic_name,
            ResolutionIdl(width=self._camera.resolution[0], height=self._camera.resolution[1])
        )

        # RGB
        self._publish(
            self._rgb_topic_name,
            RGBFrame(rgb=rgb, intrinsics=intrinsics, timestamp=np.array([now]))
        )

        # DEPTH
        self._publish(
            self._depth_topic_name,
            DepthFrame(
                depth=depth,
                # depth_image=depth_img,
                intrinsics=intrinsics,
                timestamp=np.array([now])
            )
        )


    def _teardown(self):
        pass


class RGBCameraSubscriber(RGBCamera):
    def __init__(self, resolution_topic: str, rgb_topic: str):
        super().__init__()

        self._cyclone_dp = DomainParticipant()
        self._reader_resolution = DDSReader(self._cyclone_dp, resolution_topic, ResolutionIdl)
        # Wait for the first resolution message.
        resolution = None
        while resolution is None:
            resolution = self._reader_resolution()
            logger.info("Did not yet receive resolution message. Sleeping for 1s...")
            time.sleep(1)
        self._resolution = (resolution.width, resolution.height)
        self._reader_rgb = SMReader(
            self._cyclone_dp, rgb_topic, RGBFrame.with_resolution(resolution.width, resolution.height)
        )
        # IPC stream does not currently publish fps metadata; keep a reasonable default.
        self._fps = 30

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._rgb).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._rgb

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix

    @property
    def fps(self) -> int:
        return self._fps

    def get_timestamp(self) -> float:
        """Get the timestamp of the current image."""
        return self._timestamp

    def _grab_images(self) -> None:
        frame = self._reader_rgb()
        if frame is not None:
            self._rgb = frame.rgb  # Already copied in the SMReader.
            self._intrinsics_matrix = frame.intrinsics

            self._timestamp = frame.timestamp[0].item()

class DepthCameraSubscriber(DepthCamera):
    def __init__(self, resolution_topic: str, depth_topic: str):
        super().__init__()

        self._cyclone_dp = DomainParticipant()

        # Wait for resolution first
        self._reader_resolution = DDSReader(self._cyclone_dp, resolution_topic, ResolutionIdl)
        resolution = None
        while resolution is None:
            resolution = self._reader_resolution()
            logger.info("Waiting for resolution...")
            time.sleep(1)

        # Create SHM reader for depth frames
        self._reader_depth = SMReader(
            self._cyclone_dp,
            depth_topic,
            DepthFrame.with_resolution(resolution.width, resolution.height)
        )

        self._resolution = (resolution.width, resolution.height)

    def _grab_images(self):
        frame = self._reader_depth()
        if frame is not None:
            self._depth = frame.depth
            # self._depth_image = frame.depth_image
            self._intrinsics_matrix = frame.intrinsics
            self._timestamp = frame.timestamp[0].item()

    # Implement abstract methods
    def _retrieve_depth_map(self):
        return self._depth

    def _retrieve_depth_image(self):
        return None

    @property
    def intrinsics_matrix(self):
        return self._intrinsics_matrix

    def get_timestamp(self):
        return self._timestamp

    @property
    def resolution(self):
        return self._resolution


class CameraFactory:
    def create_camera():
        # return OpenCVVideoCapture(resolution=(1920, 1080),  fps=30,intrinsics_matrix=np.eye(3))
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense

        # D405 does not support 1920x1080; try the common working profiles first.
        # candidate_resolutions = [Realsense.RESOLUTION_720, Realsense.RESOLUTION_480]
        candidate_resolutions = [Realsense.RESOLUTION_480]
        last_error = None
        for resolution in candidate_resolutions:
            try:
                logger.info(f"Trying RealSense profile: {resolution} @ 30fps")
                return Realsense(resolution=resolution, fps=30)
            except RuntimeError as error:
                last_error = error
                logger.warning(f"Failed to start RealSense with {resolution} @ 30fps: {error}")

        raise RuntimeError(
            "Could not start RealSense camera with supported profiles "
            f"{candidate_resolutions}. Last error: {last_error}"
        )


if __name__ == "__main__":
    import cv2

    initialize_ipc()

    TOPIC_RGB = "webcam_rgb"
    TOPIC_DEPTH = "webcam_depth"
    TOPIC_RESOLUTION = "webcam_resolution"
    logger.info("Creating publisher.")

    publisher = RGBCameraPublisher(CameraFactory.create_camera, TOPIC_RGB, TOPIC_DEPTH, TOPIC_RESOLUTION, 100, True)
    logger.info("Starting publisher.")
    publisher.start()

    logger.info("Creating subscriber.")
    subscriber = RGBCameraSubscriber(TOPIC_RESOLUTION, TOPIC_RGB)
    # subscriber2 = DepthCameraSubscriber(TOPIC_RESOLUTION, TOPIC_DEPTH)

    show_window = True
    try:
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    except cv2.error as error:
        show_window = False
        logger.warning(f"OpenCV GUI is not available. Running headless without display window: {error}")

    while True:
        rgb = subscriber.get_rgb_image_as_int()
        # depth = subscriber2.get_depth_map()
        image_timestamp = subscriber.get_timestamp()
        current_time = time.time()
        logger.debug(
            f"Timestamp: {image_timestamp}, Current time: {current_time}, Diff: {current_time - image_timestamp}"
        )
        rgb_cv = ImageConverter.from_numpy_int_format(rgb).image_in_opencv_format

        if show_window:
            cv2.imshow("Webcam", rgb_cv)
            key = cv2.waitKey(1)
            if key == ord("q"):
                logger.info("Stopping...")
                break
        # print(depth)
    publisher.stop()
    if show_window:
        cv2.destroyAllWindows()
