"""DDS subscriber for the bottle-cap's 3-channel voltage sensor.

Publisher side: SensorCommDDS/sensor_comm_dds/communication/readers/bottle_ble_reader.py,
publishing a `Sequence(values=[v0, v1, v2])` to topic "Bottle".
"""

import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.qos import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from sensor_comm_dds.communication.data_classes.sequence import Sequence

# Each channel's own covered/uncovered voltage separation, derived from the largest gap in 3
# recorded bottle-opening runs (see bottle_experiment/open_bottle_demo_analyze_sensors.py and
# sensor_logs/run_0000-0002.json): S0 covered in [2.25-2.86] / uncovered at [3.24], S1 covered
# in [2.69-3.12] / uncovered at [3.24-3.28], S2 covered in [2.24-2.78] / uncovered at
# [3.22-3.23]. Re-derive if the sensor mounting, cap, or opening-motion geometry changes.
PER_CHANNEL_THRESHOLDS = [3.05, 3.18, 3.00]  # S0, S1, S2, in volts

# Which sensor channels (indices into PER_CHANNEL_THRESHOLDS) must be uncovered by the end of
# each named leg of the opening motion, per the same 3 runs: S0 and S1 both pop open by leg 3,
# S2 (the last tab) by leg 5. Other legs show no reliable new transition at these thresholds
# and are deliberately not gated here.
SENSOR_CHECKPOINTS = {
    "leg_3_end": [0, 1],
    "leg_6_end": [0, 1, 2],
}


def is_uncovered(reading: np.ndarray, channels: list[int]) -> bool:
    """Whether every channel in `channels` reads above its own covered/uncovered threshold."""
    return all(reading[ch] >= PER_CHANNEL_THRESHOLDS[ch] for ch in channels)


class BottleSensorSubscriber:
    def __init__(self, topic: str = "Bottle"):
        super().__init__()
        qos = Qos(
            Policy.History.KeepLast(1),  # keep only the most recent sample
            Policy.Reliability.BestEffort,  # optional: drop samples if subscriber is too slow
        )
        self._internal_state = np.zeros(3, "float32")
        self._cyclone_dp = DomainParticipant()
        topic_bottle = Topic(self._cyclone_dp, topic, Sequence)

        self._reader = DataReader(self._cyclone_dp, topic_bottle, qos)

    def get_bottle_sensor(self) -> np.ndarray:
        samples = self._reader.take()
        if len(samples) != 0:
            self._internal_state = np.array(samples[0].values, "float32")  # update internal state
        return self._internal_state.copy()
