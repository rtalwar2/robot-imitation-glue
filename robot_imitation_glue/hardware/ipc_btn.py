import time
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from sensor_comm_dds.communication.data_classes.publishable_integer import PublishableInteger
from cyclonedds.qos import Qos, Policy
import numpy as np


class BTNSubscriber:
    def __init__(self, BTN_topic: str):
        super().__init__()
        qos = Qos(
            Policy.History.KeepLast(1),  # keep only the most recent sample
            Policy.Reliability.BestEffort,  # optional: drop samples if subscriber is too slow
        )
        self._internal_state = np.array([1],"float32")
        self._cyclone_dp = DomainParticipant()
        topic_btn = Topic(self._cyclone_dp, BTN_topic, PublishableInteger)

        self._reader = DataReader(self._cyclone_dp, topic_btn, qos)

    def get_button_state(self):
        samples = self._reader.take()
        if len(samples) != 0:
            self._internal_state = np.array([samples[0].value],"float32")  # update internal state
        return self._internal_state
