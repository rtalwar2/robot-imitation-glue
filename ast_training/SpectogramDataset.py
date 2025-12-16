import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold

class SpectrogramDataset:
    def __init__(self, spectrograms, labels, mean=None, std=None):
        """
        spectrograms: List or Array of shape (N, Time, Freq=128)
        labels: List or Array of shape (N,)
        """
        self.spectrograms = spectrograms
        self.labels = labels
        
        # AST Normalization (Crucial step from the paper)
        # If mean/std are not provided, calculate them (like get_norm_stats.py)
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
        
        # 1. Normalize: (input - mean) / (std * 2) 
        # The *2 is specific to the AST paper to bring data roughly between -0.5 and 0.5
        norm_spec = (spec - self.mean) / (self.std * 2)
        
        # 2. Convert to Tensor
        norm_spec = torch.tensor(norm_spec, dtype=torch.float32)
        
        # 3. Transpose if necessary? 
        # HuggingFace AST expects (Batch, Time, Freq). 
        # Your kaldi.fbank likely outputs (Time, Freq), which is correct.
        
        return {"input_values": norm_spec, "labels": self.labels[idx]}

# Helper to convert to HF Dataset
def create_hf_dataset(spectrograms, labels, mean, std):
    # Normalize data locally before creating HF dataset for efficiency
    # (Or use the .map function in HF datasets)
    norm_specs = [(s - mean) / (std * 2) for s in spectrograms]
    
    return Dataset.from_dict({
        "input_values": norm_specs,
        "label": labels
    })