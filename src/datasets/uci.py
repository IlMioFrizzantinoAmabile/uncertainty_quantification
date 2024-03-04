import torch
from sklearn.datasets import fetch_openml
import numpy as np
from datasets.utils import get_loader

class UCI(torch.utils.data.Dataset):
    """Some datasets from https://arxiv.org/pdf/2104.04975.pdf"""

    def __init__(
        self, 
        dataset, 
        data_path: str = "../datasets/",
        n_samples: int = None,
        seed: int = 0
    ):
        self.dataset = dataset
        self.path = data_path
        self.n_samples = n_samples
        self.data, self.targets = self.load_dataset()
        self.seed = seed
    
    def subsample(self, x, y):
        if self.n_samples is None:
            return x, y
        if self.n_samples>len(x):
            raise ValueError(f"Dataset has only {len(x)} data, you are asking for {self.n_samples}.")
        idxs = list(range(len(x)))
        np.random.seed(self.seed)
        idxs = np.random.choice(idxs, self.n_samples, replace=False)
        x = x[idxs]
        y = y[idxs]
        return x, y

    def load_dataset(self):
        if self.dataset.lower() == "boston":
            data = fetch_openml(name="Boston-house-price-data")
            x = data["data"].to_numpy()
            y = data["target"].to_numpy()
            return self.subsample(x, y)

        elif self.dataset.lower() == "concrete":
            data = fetch_openml(name="concrete_compressive_strength")
            x = data["data"].to_numpy()
            y = data["target"].to_numpy()
            return self.subsample(x, y)

        elif self.dataset.lower() == "energy":
            data = fetch_openml(name="Energy-Efficiency-Dataset")
            x = data["data"].to_numpy()[:, :-2]
            y = data["data"].to_numpy()[:, -2:]
            return self.subsample(x, y)

        elif self.dataset.lower() == "kin8nm":
            data = fetch_openml(name="kin8nm")
            x = data["data"].to_numpy()
            y = data["target"].to_numpy()
            return self.subsample(x, y)

        elif self.dataset.lower() == "wine":
            data = fetch_openml(name="WineDataset")
            x = data["data"].to_numpy()[:, :-1]
            y = data["data"].to_numpy()[:, -1:]
            return self.subsample(x, y)

        elif self.dataset.lower() == "yacht":
            data = fetch_openml(name="yacht_hydrodynamics")
            x = data["data"].to_numpy()
            y = data["target"].to_numpy()
            return self.subsample(x, y)


    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)
    


def get_uci_loaders(
        uci_type,
        batch_size = 128,
        train_val_ratio: float = 0.9,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../datasets",
    ):
    dataset = UCI(uci_type, data_path=data_path, n_samples=None, seed=seed)
    train_loader, valid_loader = get_loader(
        dataset,
        batch_size=batch_size,
        train_val_ratio=train_val_ratio,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, None
