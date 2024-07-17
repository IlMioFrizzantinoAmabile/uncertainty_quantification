from src.datasets.sinusoidal import Sinusoidal, get_sinusoidal
from src.datasets.mnist import MNIST, get_mnist, get_rotated_mnist
from src.datasets.emnist import EMNIST, get_emnist, get_rotated_emnist
from src.datasets.kmnist import KMNIST, get_kmnist, get_rotated_kmnist
from src.datasets.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.datasets.cifar10 import CIFAR10, get_cifar10, get_cifar10_augmented, get_cifar10_corrupted
from src.datasets.cifar100 import CIFAR100, get_cifar100, get_cifar100_augmented
from src.datasets.svhn import SVHN, get_svhn, get_svhn_augmented
from src.datasets.food101 import FOOD101, get_food101_scaled

#from src.datasets.all_datasets import get_train_loaders, get_test_loaders
from src.datasets.wrapper import dataloader_from_string, augmented_dataloader_from_string
from src.datasets.utils import get_output_dim