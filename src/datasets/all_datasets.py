import torch

from src.datasets.sinusoidal import Sinusoidal, get_sinusoidal
from src.datasets.mnist import MNIST, get_mnist, get_rotated_mnist
from src.datasets.emnist import EMNIST, get_emnist, get_rotated_emnist
from src.datasets.kmnist import KMNIST, get_kmnist, get_rotated_kmnist
from src.datasets.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.datasets.cifar10 import CIFAR10, get_cifar10, get_cifar10_augmented, get_cifar10_corrupted
from src.datasets.cifar100 import CIFAR100, get_cifar100, get_cifar100_augmented
from src.datasets.svhn import SVHN, get_svhn, get_svhn_augmented
from src.datasets.food101 import FOOD101, get_food101_scaled
from src.datasets.celeba import CelebA, get_celeba, get_celeba_augmented, get_celeba_ood

def get_train_loaders(
        dataset_name,
        n_samples = None,
        batch_size: int = 128,
        shuffle = True,
        seed: int = 0,
        download: bool = True,
        data_path: str = "../datasets",
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if dataset_name == "Sinusoidal":
        train_loader, valid_loader, test_loader = get_sinusoidal(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples = n_samples,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "MNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_mnist(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "FMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_fmnist(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-10":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_cifar10_augmented(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-100":
        classes = list(range(100))
        train_loader, valid_loader, test_loader = get_cifar100_augmented(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/100) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "SVHN":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_svhn_augmented(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CelebA":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_celeba_augmented(
            batch_size = batch_size, 
            shuffle = shuffle,
            seed = seed,
            download = False, 
            data_path = data_path
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")
    
    return train_loader, valid_loader


def get_test_loaders(
        dataset_name,
        n_samples = None,
        batch_size: int = 128,
        shuffle = True,
        seed: int = 0,
        download: bool = False,
        data_path: str = "../datasets",
        angle: float = 0,               # for rotated datasets
        corr_type: str = "fog",         # for corrupted datasets
        severity_level: int = 5,        # for corrupted datasets
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if dataset_name.startswith("MNIST-R"):
        angle = int(dataset_name.removeprefix("MNIST-R"))
        dataset_name = "MNIST-R"
    elif dataset_name.startswith("FMNIST-R"):
        angle = int(dataset_name.removeprefix("FMNIST-R"))
        dataset_name = "FMNIST-R"
    elif dataset_name.startswith("CIFAR-10-C"):
        severity_level, corr_type = dataset_name.removeprefix("CIFAR-10-C").split('-')
        severity_level = int(severity_level)
        dataset_name = "CIFAR-10-C"
        
    if dataset_name == "Sinusoidal":
        train_loader, valid_loader, test_loader = get_sinusoidal(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples = n_samples,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "MNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_mnist(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "MNIST-R":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_rotated_mnist(
            angle = angle,
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "FMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_fmnist(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "FMNIST-R":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_rotated_fmnist(
            angle = angle,
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "EMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_emnist(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "KMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_kmnist(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-10":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_cifar10(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-10-C":
        classes = list(range(10))
        # train and valid are None
        train_loader, valid_loader, test_loader = get_cifar10_corrupted(
            corr_type = corr_type,
            severity_level = severity_level,
            batch_size = batch_size, 
            shuffle = shuffle,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-100":
        classes = list(range(100))
        train_loader, valid_loader, test_loader = get_cifar100(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/100) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "SVHN":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_svhn(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "FOOD101":
        train_loader, valid_loader, test_loader = get_food101_scaled(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = None,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CelebA":
        train_loader, valid_loader, test_loader = get_celeba(
            batch_size = batch_size, 
            shuffle = shuffle,
            seed = seed,
            download = False, 
            data_path = data_path
        )
    elif dataset_name.startswith("CelebA-"):
        only_with = dataset_name.removeprefix("CelebA-")
        print(f"Loading CelebA only with {only_with}")
        train_loader, valid_loader, test_loader = get_celeba_ood(
            only_with = only_with,
            batch_size = batch_size, 
            shuffle = shuffle,
            seed = seed,
            download = False, 
            data_path = data_path
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")
    
    return train_loader, valid_loader, test_loader