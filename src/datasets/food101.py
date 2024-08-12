from pathlib import Path
import torch
import torchvision
import numpy as np
from src.datasets.utils import get_loader, get_subset_data

class FOOD101(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        transform = None,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed: int = 0,
        download: bool = True,
        data_path: str = "../datasets",
    ):
        self.transform = transform
        self.path = Path(data_path)
        self.dataset = torchvision.datasets.Food101(root=self.path, split='train' if train else 'test', download=download)

    def __getitem__(self, index):
        #img, target = self.data[index], self.targets[index]
        img, target = self.dataset.__getitem__(index)
        img = np.asarray(img)
        if self.transform is not None:
            # torch wants channel dimension before height and width
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transform(img)
            img = img.permute(1, 2, 0).numpy()
        return img, target

    def __len__(self):
        return 10000 #len(self.dataset)
    

def get_food101_scaled(
        batch_size = 128,
        shuffle = False,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed = 0,
        download: bool = True,
        data_path="../datasets",
        shape = (218, 178)
    ):
    augment_transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomResizedCrop((218, 178),scale=(1.,1.0),ratio=(1.,1.), antialias=True),
            torchvision.transforms.Resize(shape),
            torchvision.transforms.Normalize(
                mean=[142.21465, 111.33642,  87.96465], std=[67.121956, 69.36996,  72.18046 ]
            ),
            ])
    dataset = FOOD101(
        train=True,
        transform=augment_transform,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = FOOD101(
        train=False,
        transform=augment_transform,
        n_samples_per_class=None,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
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