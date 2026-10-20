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

# Each channel's covered/uncovered threshold, re-derived 2026-10-09 for the retuned motion (6 legs,
# gripper turning with the path) from the "test" split runs whose cap was fully closed at the grip point:
# run_0024, 0026-0028, 0030, 0032-0038 (runs 0025/0029/0031 started with S2 already reading open -- its
# lid section had not latched -- and are excluded). Measured per channel as the level held MORE THAN 1 s
# before its final jump vs the level after it, because that is what separates closed from open; the
# tool's raw covered/open bands count the partly-lifted stretch before the jump as "covered":
#   S0 3.20 -- 3.17-3.22 V before vs 3.23-3.24 V after: no usable gap, and it crosses back and forth
#              1-6 times per run. It always opens FIRST (push / leg 1), so it is kept for logging and
#              for CALIBRATED_RANGE but decides nothing (see OPEN_CHANNELS, SENSOR_CHECKPOINTS).
#   S1 3.12 -- 2.98-3.02 V before vs 3.23-3.26 V after (~0.2 V gap), one clean jump during leg 2.
#   S2 3.10 -- 2.94-2.95 V before vs 3.26 V after (~0.3 V gap). It sits partly lifted (2.7-2.95 V)
#              through the legs and jumps during the last ones; it is the LAST channel to open, so
#              "S2 open" is "the cap is fully open".
# This file is the single source: the analyze fork, the collector and eval all import these, so
# the only value that must be re-derived alongside them is CALIBRATED_RANGE in train_ast_bottle.py.
PER_CHANNEL_THRESHOLDS = [3.20, 3.12, 3.10]  # S0, S1, S2, in volts

# The channels that decide "the cap is open" (stop-on-open, episode success): S1 and S2. S0 is left
# out because it cannot separate the two states (above) and always opens before the others anyway.
OPEN_CHANNELS = [1, 2]

# Which sensor channel must be uncovered by each named checkpoint (event = end of that stage; the legs
# are numbered leg_2 .. leg_7 for the six LEGS entries). From the 2026-10-09 runs: S1 jumps 2-3 s before
# leg_3_end in every run; S2 jumps 1-4 s AFTER leg_5_end -- end of leg 5 / during leg 6 -- so it is
# judged at the end of the last leg, leg_7_end (gating it at leg_5_end, as before the retune, would fail
# every run). S0 is not gated. Re-check this map whenever LEGS changes length.
SENSOR_CHECKPOINTS = {
    "leg_3_end": [1],
    "leg_7_end": [2],
}


def is_uncovered(reading: np.ndarray, channels: list[int]) -> bool:
    """Whether every channel in `channels` reads above its own covered/uncovered threshold."""
    return all(reading[ch] >= PER_CHANNEL_THRESHOLDS[ch] for ch in channels)


def is_cap_open(reading: np.ndarray) -> bool:
    """Whether the deciding channels (OPEN_CHANNELS) all read open."""
    return is_uncovered(reading, OPEN_CHANNELS)


def channels_open(reading: np.ndarray) -> list[int]:
    """The deciding channels (OPEN_CHANNELS) that already read open -- at the grip point, before the push,
    these cannot report an opening: test pose 2 of 2026-10-09 (run_0025) had S2 at 3.27 V from the start
    (its lid section had not latched), so stop-on-open fired as soon as S0 and S1 really opened and the cap
    was left half closed."""
    return [ch for ch in OPEN_CHANNELS if reading[ch] >= PER_CHANNEL_THRESHOLDS[ch]]


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
