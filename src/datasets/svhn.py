from pathlib import Path
import torch
import torchvision
from src.datasets.utils import get_loader, get_subset_data

class SVHN(torch.utils.data.Dataset):
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
        self.dataset = torchvision.datasets.SVHN(root=self.path, split='train' if train else 'test', download=download)

        if len(classes)>=10 and n_samples_per_class is None:
            self.data, self.targets = self.dataset.data, self.dataset.labels
        else:
            self.data, self.targets = get_subset_data(self.dataset.data, self.dataset.labels, classes, n_samples_per_class=n_samples_per_class, seed=seed)

        mean, std = [x*255 for x in (0.485, 0.456, 0.406)], [x*255 for x in (0.229, 0.224, 0.225)]
        self.data = torchvision.transforms.functional.normalize(
            torch.from_numpy(self.data).float(),
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
    

def get_svhn(
        batch_size = 128,
        shuffle = False,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed = 0,
        download: bool = True,
        data_path="../datasets",
    ):
    dataset = SVHN(
        train=True,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = SVHN(
        train=False,
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


def get_svhn_augmented(
        batch_size = 128,
        shuffle = False,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed = 0,
        download: bool = True,
        data_path="../datasets",
    ):
    augment_transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1), antialias=True),
            ])
    dataset = SVHN(
        train=True,
        transform=augment_transform,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = SVHN(
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