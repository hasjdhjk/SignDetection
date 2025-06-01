#
# 
# This file contains the LandmarkDataset class required for PyTorch. 
# Please check if things are ok xx
#
#

import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset

class LandmarkDataset(Dataset):
    def __init__(self, root_dir, sequence_length=45, label_map_path="label_map.json"):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.label_map = {}
        label_counter = 0

        # Traverse label subdirectories
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue

            if label not in self.label_map:
                self.label_map[label] = label_counter
                label_counter += 1

            for file in os.listdir(label_path):
                if file.endswith(".npy"):
                    filepath = os.path.join(label_path, file)
                    sequence = np.load(filepath)

                    # Optional: normalize timestamps to [0, 1]
                    timestamps = sequence[:, -1]
                    timestamps = (timestamps - timestamps.min()) / (timestamps.ptp() + 1e-6)
                    sequence[:, -1] = timestamps

                    # Pad or trim to fixed length
                    if len(sequence) < self.sequence_length:
                        pad_len = self.sequence_length - len(sequence)
                        padding = np.zeros((pad_len, sequence.shape[1]))
                        sequence = np.vstack([sequence, padding])
                    else:
                        sequence = sequence[:self.sequence_length]

                    self.data.append(sequence)
                    self.labels.append(self.label_map[label])

        # Save label map
        with open(label_map_path, "w") as f:
            json.dump(self.label_map, f)

        # Convert to tensors
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
