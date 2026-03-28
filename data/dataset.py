"""
Dataset loader for Nimbus-1 training.
Reads pre-tokenized binary files as memory-mapped numpy arrays.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = torch.from_numpy(
            self.data[idx : idx + self.block_size + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_dataloader(data_path: str, block_size: int, batch_size: int,
                   num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    dataset = TokenDataset(data_path, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
