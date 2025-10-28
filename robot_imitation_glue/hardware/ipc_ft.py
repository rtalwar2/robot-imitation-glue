"""This file implements a multiprocess pub-sub for an airo-mono RGB camera.

This requires you to install the airo-camera-toolkit, which you can do by following the instructions here:
https://github.com/airo-ugent/airo-mono
"""


import time
import numpy as np
from cyclonedds.domain import DomainParticipant
from sensor_comm_dds.communication.data_classes.sequence import Sequence
from sensor_comm_dds.communication.readers.ft_reader import FTReaderConfig
from cyclonedds.qos import Qos, Policy
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from airo_ipc.framework.node import Node
from airo_ipc.framework.framework import IpcKind
from rtde_receive import RTDEReceiveInterface

from loguru import logger

class FTPublisher(Node):
    """The publisher will read FT and publish in a loop"""

    def __init__(self,config: FTReaderConfig, update_frequency=1, verbose=False):
        self.config = config
        self._ft_topic_name = "FT"
        max_connection_attempts = 3
        for connection_attempt in range(max_connection_attempts):
            try:
                self.rtde_receive = RTDEReceiveInterface(self.config.ROBOT_IP)
                break
            except RuntimeError as e:
                logger.warning("Failed to connect to RTDE, retrying...")
                if connection_attempt == max_connection_attempts:
                    raise RuntimeError("Could not connect to RTDE. Is the robot in remote control? Is the IP correct? Is the network ok?")
                else:
                    time.sleep(1)
                    
        super().__init__(update_frequency, verbose)

    def _setup(self):

        logger.info("Registering FT publishers.")
        self._register_publisher(self._ft_topic_name, Sequence,IpcKind.DDS)

    def _step(self):
        """The _step method is called in a loop by the Node superclass."""

        sample = Sequence([0 for _ in range(6)])
        sample.values = self.rtde_receive.getActualTCPForce()
        logger.debug(sample.values)
        self._publish(
            self._ft_topic_name,sample)
        
    def _teardown(self):
        pass


# class FTPublisher:
#     """The publisher will open the webcam and publish the resolution and frame in a loop."""

#     def __init__(self):
#         self._ft_reader = FTReader(
#             config=FTReaderConfig(ENABLE_WS=False, topic_names=np.array(["FT"]))
#         )

#     def run(self):
#         self._ft_reader.run()


class FTSubscriber:
    def __init__(self, FT_topic: str):
        super().__init__()
        qos = Qos(
            Policy.History.KeepLast(1),  # keep only the most recent sample
            Policy.Reliability.BestEffort,  # optional: drop samples if subscriber is too slow
        )

        self._cyclone_dp = DomainParticipant()
        topic_FT = Topic(self._cyclone_dp, FT_topic, Sequence)

        self._reader = DataReader(self._cyclone_dp, topic_FT, qos)

    def get_FT(self):
        return np.array(self._reader.take()[0].values,"float32")
