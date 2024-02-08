from pathlib import Path
import torch
import torchvision
from src.datasets.utils import get_loader, get_subset_data

class CIFAR100(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        transform = None,
        n_samples_per_class: int = None,
        classes: list = list(range(100)),
        seed: int = 0,
        download: bool = True,
        data_path: str = "../datasets",
    ):
        self.transform = transform
        self.path = Path(data_path)
        self.dataset = torchvision.datasets.CIFAR100(root=self.path, train=train, download=download)

        if len(classes)>=100 and n_samples_per_class is None:
            self.data, self.targets = self.dataset.data, self.dataset.targets
        else:
            self.data, self.targets = get_subset_data(self.dataset.data, self.dataset.targets, classes, n_samples_per_class=n_samples_per_class, seed=seed)

        mean, std = [x*255 for x in (0.4914, 0.4822, 0.4465)], [x*255 for x in (0.2470, 0.2435, 0.2616)]
        self.data = torchvision.transforms.functional.normalize(
            torch.from_numpy(self.data).float().permute(0, 3, 1, 2),
            mean,
            std
        ).permute(0, 2, 3, 1).numpy()
        self.targets = torch.nn.functional.one_hot(torch.tensor(self.targets), len(classes)).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            # torch wants channel dimension before height and width
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = self.transform(img)
            img = img.permute(1, 2, 0).numpy()
        return img, target

    def __len__(self):
        return len(self.data)
    

def get_cifar100(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../datasets",
    ):
    dataset = CIFAR100(
        train=True,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = CIFAR100(
        train=False,
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


def get_cifar100_augmented(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../datasets",
    ):
    train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1), antialias=True),
            ])
    dataset = CIFAR100(
        train=True,
        transform=train_transform,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = CIFAR100(
        train=False,
        transform=None,
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