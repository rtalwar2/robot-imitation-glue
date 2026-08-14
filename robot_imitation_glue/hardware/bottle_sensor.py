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

# Each channel's own covered/uncovered voltage separation, derived from runs 0003/0005/0006
# (the 3 runs under the current motion parameters -- 0000-0002 used a different sensor
# layout and aren't comparable). S0 covered ~3.01-3.11 / uncovered ~3.24 (jumps at the
# initial push, before any leg) -- covered baseline has drifted up across these runs, so the
# threshold sits above the highest observed covered reading (3.11) with only ~0.13V margin
# to the lowest uncovered reading (3.24); re-check if it drifts further. S1 covered ~2.9-3.14
# / uncovered ~3.28 (jumps at leg_3). S2 covered ~2.6-2.9 / uncovered ~3.23-3.24 (jumps by
# leg_5 in 2/3 runs, needed one retry to clear by leg_6 in the third). Re-derive if the
# sensor mounting, cap, or opening-motion geometry changes.
PER_CHANNEL_THRESHOLDS = [3.17, 3.18, 3.00]  # S0, S1, S2, in volts

# Which sensor channel (index into PER_CHANNEL_THRESHOLDS) must be uncovered by each named
# checkpoint of the opening motion, per run_0003.json: S0 pops open on the initial push
# (before any leg), S1 by leg_3, S2 by leg_6. Each checkpoint gates exactly the one channel
# that reliably transitions there -- earlier checkpoints don't re-check channels that already
# passed, since a sensor covering back up isn't expected once uncovered.
SENSOR_CHECKPOINTS = {
    "push_end": [0],
    "leg_3_end": [1],
    "leg_6_end": [2],
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
        self._has_received_sample = False
        self._cyclone_dp = DomainParticipant()
        topic_bottle = Topic(self._cyclone_dp, topic, Sequence)

        self._reader = DataReader(self._cyclone_dp, topic_bottle, qos)

    def get_bottle_sensor(self) -> np.ndarray:
        samples = self._reader.take()
        if len(samples) != 0:
            self._internal_state = np.array(samples[0].values, "float32")  # update internal state
            self._has_received_sample = True
        return self._internal_state.copy()

    def has_received_sample(self) -> bool:
        """False until the first real DDS sample arrives -- distinguishes a genuine all-zero
        reading from get_bottle_sensor()'s cold-start default (which a dead/disconnected
        bottle_ble_reader.py would leave in place forever)."""
        return self._has_received_sample
