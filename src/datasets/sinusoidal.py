import torch
import numpy as np
from src.datasets.utils import get_loader

class Sinusoidal(torch.utils.data.Dataset):
    def __init__(
        self,
        intervals = [(-1,0.),(1., 2.)],
        train: bool = True,
        n_samples: int = None,
        seed: int = 0
    ):
        noise = 0.1
        if train:
            np.random.seed(seed)
        else:
            np.random.seed(seed+1000)
        n_samples = 1000 if n_samples is None else n_samples
        interval_size = int(n_samples / len(intervals))
        self.data = np.concatenate([
            np.random.uniform(low=low, high=high, size=interval_size) for low,high in intervals
        ])
        #self.data = 2*np.random.random(n_samples) - 1
        self.targets = np.sin(2*np.pi*self.data) + noise * np.random.randn(n_samples)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)


def get_sinusoidal(
        batch_size = 128,
        train_val_ratio: float = 0.9,
        n_samples : int = None,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../datasets",
    ):
    dataset = Sinusoidal(
        intervals=((-0.75,0), (0.5,1.5)),
        train=True,
        n_samples=n_samples,
        seed=seed
    )
    dataset_test = Sinusoidal(
        intervals=((-1,0.25), (0.25,1.75)),
        train=False,
        n_samples=n_samples,
        seed=seed
    )
    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    test_loader = get_loader(
        dataset_test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, test_loader