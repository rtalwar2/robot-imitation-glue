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

# Each channel's covered/uncovered separation, derived by bottle_experiment/derive_thresholds.py
# from the calibration runs recorded under the CURRENT motion: run_0013, run_0018, run_0020-0023
# (the "test" split poses). run_0017/run_0019 are pose 1, which never opened, and the tool
# excludes them by itself. Worst-case bands over those runs: S0 covered p99 3.103 / open p1 3.213,
# S1 2.975 / 3.204, S2 2.932 / 3.223 V.
#   S0 3.16 -- midpoint of a deliberately thin band: S0's whole excursion is only ~0.11-0.16 V, so
#              any threshold leaves ~0.05 V of cushion either side. Re-check this channel whenever
#              the mounting is touched.
#   S1 3.12 -- biased high on purpose. The old 3.18 sat only 0.024 V below the observed open floor,
#              so a genuine open could read covered; 3.12 keeps 0.145 V above the covered ceiling
#              and 0.084 V below the open floor. (The tool's symmetric midpoint is 3.09.)
#   S2 3.08 -- midpoint. The old 3.00 sat only 0.068 V above the covered ceiling, which is the
#              dangerous direction: a still-covered cap could be declared open.
# This file is the single source: the analyze fork, the collector and eval all import these, so
# the only value that must be re-derived alongside them is CALIBRATED_RANGE in train_ast_bottle.py.
# Run derive_thresholds.py on the calibration batch to move the two together.
PER_CHANNEL_THRESHOLDS = [3.16, 3.12, 3.08]  # S0, S1, S2, in volts

# Which sensor channel (index into PER_CHANNEL_THRESHOLDS) must be uncovered by each named
# checkpoint, measured on the same runs. S0 crosses right at the end of the push (-0.26 s to
# +0.12 s around push_end) and was at 100% within the first 16% of leg 2 in every run, so gating it
# AT push_end failed 2 of 4 runs that opened perfectly -- and with the retry mechanism that means a
# needless retract + DEPTH_NUDGE_M deeper. It now gates at leg_2_end. S1 and S2 both cross during
# leg 3 (S1 ~36-49% and S2 ~66-83% into the segment), so leg_3_end gates both. leg_6_end is gone:
# S2 had already been open for ~2.4 s by then, so that checkpoint passed trivially and proved
# nothing.
SENSOR_CHECKPOINTS = {
    "leg_2_end": [0],
    "leg_3_end": [1, 2],
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
