import os
import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.util import duration
from loguru import logger
from cyclonedds.qos import Qos, Policy

from sensor_comm_dds.communication.data_classes.sequence import Sequence

import matplotlib.cm as cm

class SpectrogramSubscriber:
    def __init__(self, topic_name="MelSpectrogram"):
        qos = Qos(
            Policy.History.KeepLast(1),  # keep only the most recent sample
            Policy.Reliability.BestEffort,  # optional: drop samples if subscriber is too slow
        )

        # DDS setup
        self.participant = DomainParticipant()
        self.topic = Topic(self.participant, topic_name, Sequence)
        self.reader = DataReader(self.participant, self.topic,qos)
        logger.info(f"Subscribed to DDS topic: {topic_name}")
        self.prev_sample=None
    # def get_spectogram(self): grayscale with 3 channels
    #     sample = self.reader.take()
    #     if len(sample)==0:
    #         return self.prev_sample
    #     sample=sample[0]
    #     # Parse the Sequence
    #     n_mels = int(sample.values[0])
    #     n_frames = int(sample.values[1])
    #     flat_data = np.array(sample.values[2:], dtype=float)
    #     mel_spectrogram_db = flat_data.reshape((n_mels, n_frames,1))

    #     # --- Rescale from [min, max] to [0, 255] ---
    #     min_val = 50
    #     max_val = 130
    #     scaled = (mel_spectrogram_db - min_val) / (max_val - min_val) * 255.0
    #     scaled = np.clip(scaled, 0, 255).astype(np.uint8)

    #     three_channel = np.repeat(scaled, 3, axis=2) 
    #     self.prev_sample=three_channel
    #     return three_channel



    def get_spectogram(self):
        sample = self.reader.take()
        if len(sample) == 0:
            return self.prev_sample

        sample = sample[0]


        # Parse the Sequence
        n_mels = int(sample.values[0])
        n_frames = int(sample.values[1])
        flat_data = np.array(sample.values[2:], dtype=float)
        mel_spectrogram_db = flat_data.reshape((n_mels, n_frames))

        # --- Normalize to 0–1 for colormap ---
        min_val = 50
        max_val = 130
        normalized = (mel_spectrogram_db - min_val) / (max_val - min_val + 1e-8)

        # --- Apply colormap ---
        cmap = cm.get_cmap('viridis')  # you can use 'magma', 'plasma', 'inferno', etc.
        colored = cmap(normalized)     # returns an RGBA array (values in 0–1)

        # --- Convert RGBA (float 0–1) to RGB uint8 (0–255) ---
        rgb_image = (colored[..., :3] * 255).astype(np.uint8)

        self.prev_sample = rgb_image
        return rgb_image

# if __name__ == "__main__":
#     logger.info(f"Running {os.path.basename(__file__)}")
#     subscriber = SpectrogramSubscriber(topic_name="MelSpectrogram")
#     subscriber.run()




class SpectrogramSubscriberKaldi:
    def __init__(self, topic_name="KaldiSpectrogram"):
        qos = Qos(
            Policy.History.KeepLast(1),  # keep only the most recent sample
            Policy.Reliability.BestEffort,  # optional: drop samples if subscriber is too slow
        )

        # DDS setup
        self.participant = DomainParticipant()
        self.topic = Topic(self.participant, topic_name, Sequence)
        self.reader = DataReader(self.participant, self.topic,qos)
        logger.info(f"Subscribed to DDS topic: {topic_name}")
        self.prev_sample=None
    # def get_spectogram(self): grayscale with 3 channels
    #     sample = self.reader.take()
    #     if len(sample)==0:
    #         return self.prev_sample
    #     sample=sample[0]
    #     # Parse the Sequence
    #     n_mels = int(sample.values[0])
    #     n_frames = int(sample.values[1])
    #     flat_data = np.array(sample.values[2:], dtype=float)
    #     mel_spectrogram_db = flat_data.reshape((n_mels, n_frames,1))

    #     # --- Rescale from [min, max] to [0, 255] ---
    #     min_val = 50
    #     max_val = 130
    #     scaled = (mel_spectrogram_db - min_val) / (max_val - min_val) * 255.0
    #     scaled = np.clip(scaled, 0, 255).astype(np.uint8)

    #     three_channel = np.repeat(scaled, 3, axis=2) 
    #     self.prev_sample=three_channel
    #     return three_channel



    def get_spectogram(self):
        sample = self.reader.take()
        if len(sample) == 0:
            return self.prev_sample

        sample = sample[0]
        # Parse the Sequence
        n_mels = int(sample.values[0])
        n_frames = int(sample.values[1])
        flat_data = np.array(sample.values[2:], dtype=np.float32)
        mel_spectrogram_db = flat_data.reshape((n_mels, n_frames)) #192,128
        # --- Normalize to 0–1 for colormap ---
        min_val = -20
        max_val = 40
        normalized = (mel_spectrogram_db - min_val) / (max_val - min_val)

        # --- Apply colormap ---
        cmap = cm.get_cmap('viridis')  # you can use 'magma', 'plasma', 'inferno', etc.
        colored = cmap(np.transpose(normalized,(1,0)))     # returns an RGBA array (values in 0–1)

        # --- Convert RGBA (float 0–1) to RGB uint8 (0–255) ---
        rgb_image = (colored[..., :3] * 255).astype(np.uint8)

        grayscale_data_image = np.repeat(normalized.reshape((n_mels, n_frames,1)), 3, axis=2) 
        self.prev_sample = rgb_image,grayscale_data_image

        return self.prev_sample

# if __name__ == "__main__":
#     logger.info(f"Running {os.path.basename(__file__)}")
#     subscriber = SpectrogramSubscriber(topic_name="MelSpectrogram")
#     subscriber.run()
