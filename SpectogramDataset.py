import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from datasets import Dataset

# This is the class we defined in the first step
class SpectrogramDataset:
    def __init__(self, spectrograms, labels, mean=None, std=None):
        self.spectrograms = spectrograms
        self.labels = labels
        
        # Calculate stats if not provided (Crucial for AST normalization)
        if mean is None or std is None:
            self.mean = np.mean(spectrograms)
            self.std = np.std(spectrograms)
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        
        # AST Normalization Formula: (input - mean) / (std * 2)
        norm_spec = (spec - self.mean) / (self.std * 2)
        
        # Convert to Tensor (Time, Freq)
        norm_spec = torch.tensor(norm_spec, dtype=torch.float32)
        
        return {"input_values": norm_spec, "labels": self.labels[idx]}