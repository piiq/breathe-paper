"""Audio Dataset loader.

This file includes utilities and helper functions from the repository that contains
the PyTorch code for the following paper:
[Rethinking CNN Models for Audio Classification](https://arxiv.org/abs/2007.11154)

In the original project the classes and methods were located in different files and
folders so simple copying and pasting code from this section into the original
project might not work.
"""
import pickle as pkl  # nosec

import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """Audio dataset object."""

    def __init__(self, pkl_dir, dataset_name, transforms=None):
        self.data = []
        self.length = 1500 if dataset_name == "GTZAN" else 250
        self.transforms = transforms
        with open(pkl_dir, "rb") as f:
            self.data = pkl.load(f)

    def __len__(self):
        """Get data length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item."""
        entry = self.data[idx]
        values = entry["values"].reshape(-1, 128, self.length)
        values = torch.Tensor(values)
        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry["target"]])  # pylint: disable=no-member
        return (values, target)


def fetch_dataloader(
    pkl_dir: str, dataset_name: str, batch_size: int, num_workers: int
) -> DataLoader:
    """Fetch a dataloader.

    Parameters
    ----------
    pkl_dir : str
        Path to the pickle folder
    dataset_name : str
        Name of the dataset
    batch_size : int
        Size of the batch
    num_workers : int
        Number of workers

    Returns
    -------
    DataLoader
        Torch dataloader
    """
    dataset = AudioDataset(pkl_dir, dataset_name)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    return dataloader
