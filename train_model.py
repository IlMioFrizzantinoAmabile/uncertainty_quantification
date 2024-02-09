import pickle
import os
import argparse
import json
import datetime
from flax import linen as nn

from src.datasets.mnist import MNIST, get_mnist, get_rotated_mnist
from src.datasets.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.datasets.cifar10 import CIFAR10, get_cifar10, get_cifar10_augmented, get_cifar10_corrupted
from src.datasets.cifar100 import CIFAR100, get_cifar100, get_cifar100_augmented
from src.datasets.svhn import SVHN, get_svhn, get_svhn_augmented
from src.datasets.celeba import CelebA, get_celeba, get_celeba_augmented
from src.datasets.utils import get_output_dim

from src.models import MLP, LeNet,ResNet, ResNetBlock, GoogleNet

from src.training.minimizer import maximum_a_posteriori



parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--uci_type", type=str, choices=["concrete", "boston", "energy", "kin8nm", "wine", "yacht"], default=None)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet", "ResNet", "GoogleNet"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu"], default="tanh", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=2, type=int, help="Number of layers in the MLP.")

# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "rmsprop"], default="adam")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification"], default="classification")
parser.add_argument("--default_train_hp", action="store_true", required=False, default=False)

# storage
parser.add_argument("--run_name", default=None, help="Fix the save file name. If None it's set to starting time")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--test_every_n_epoch", type=int, default=1e10, help="Frequency of coputing validation stats")



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


    ###############
    ### dataset ###
    if args.dataset == "MNIST":
        classes = list(range(10))
        train_loader, valid_loader, _ = get_mnist(
            batch_size = args.batch_size, 
            shuffle = True,
            n_samples_per_class = int(args.n_samples/10) if args.n_samples is not None else None,
            classes = classes,
            seed = args.seed,
            download = True, 
            data_path = args.data_path
        )
    elif args.dataset == "FMNIST":
        classes = list(range(10))
        train_loader, valid_loader, _ = get_fmnist(
            batch_size = args.batch_size, 
            shuffle = True,
            n_samples_per_class = int(args.n_samples/10) if args.n_samples is not None else None,
            classes = classes,
            seed = args.seed,
            download = True, 
            data_path = args.data_path
        )
    elif args.dataset == "CIFAR-10":
        classes = list(range(10))
        train_loader, valid_loader, _ = get_cifar10_augmented(
            batch_size = args.batch_size, 
            shuffle = True,
            n_samples_per_class = int(args.n_samples/10) if args.n_samples is not None else None,
            classes = classes,
            seed = args.seed,
            download = True, 
            data_path = args.data_path
        )
    elif args.dataset == "CIFAR-100":
        classes = list(range(100))
        train_loader, valid_loader, _ = get_cifar100_augmented(
            batch_size = args.batch_size, 
            shuffle = True,
            n_samples_per_class = int(args.n_samples/100) if args.n_samples is not None else None,
            classes = classes,
            seed = args.seed,
            download = True, 
            data_path = args.data_path
        )
    elif args.dataset == "SVHN":
        classes = list(range(10))
        train_loader, valid_loader, _ = get_svhn_augmented(
            batch_size = args.batch_size, 
            shuffle = True,
            n_samples_per_class = int(args.n_samples/10) if args.n_samples is not None else None,
            classes = classes,
            seed = args.seed,
            download = True, 
            data_path = args.data_path
        )
    elif args.dataset == "CelebA":
        classes = list(range(10))
        train_loader, valid_loader, _ = get_celeba(
            batch_size = args.batch_size, 
            shuffle = True,
            seed = args.seed,
            download = False, 
            data_path = args.data_path
        )
    else:
        raise ValueError(f"Dataset {args.dataset} is not implemented")
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}")


    #############
    ### model ###
    output_dim = get_output_dim(args.dataset)
    act_fn = getattr(nn, args.activation_fun)

    have_batch_stats = False
    if args.model == "MLP":
        model = MLP(
            output_dim = output_dim, 
            num_layers = args.mlp_num_layers,
            hidden_dim = args.mlp_hidden_dim, 
            act_fn = act_fn
        )
        if args.default_train_hp:
            args_dict["n_epochs"] = 50
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "adam"
            args_dict["learning_rate"] = 1e-3,
            args_dict["momentum"] = None
            args_dict["weight_decay"] = None
    elif args.model == "LeNet":
        model = LeNet(
            output_dim = output_dim, 
            act_fn = act_fn
        )
        if args.default_train_hp:
            args_dict["n_epochs"] = 50
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "adam"
            args_dict["learning_rate"] = 1e-3,
            args_dict["momentum"] = None
            args_dict["weight_decay"] = None
    elif args.model == "ResNet":
        model = ResNet(
            output_dim = output_dim,
            c_hidden =(16, 32, 64),
            num_blocks = (3, 3, 3),
            act_fn = act_fn,
            block_class = ResNetBlock
        )
        if args.default_train_hp:
            args_dict["n_epochs"] = 200
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "sgd"
            args_dict["learning_rate"] = 0.1
            args_dict["momentum"] = 0.9
            args_dict["weight_decay"] = 1e-4
    elif args.model == "GoogleNet":
        model = GoogleNet(
            output_dim = output_dim,
            act_fn = act_fn
        )
        if args.default_train_hp:
            args_dict["n_epochs"] = 200
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "adamw"
            args_dict["learning_rate"] = 1e-3,
            args_dict["momentum"] = None
            args_dict["weight_decay"] = 1e-4
    else:
        raise ValueError(f"Model {args.model} is not implemented")
    args_dict["opt_hp"] = {
            "lr": args_dict["learning_rate"],
            "momentum": args_dict["momentum"],
            "weight_decay": args_dict["weight_decay"],
        }


    ################
    ### training ###  
    params, batch_stats, stats_dict = maximum_a_posteriori(
            model, train_loader, valid_loader, args_dict
        )
    model_dict = {
        "params": params,
        "batch_stats": batch_stats,
        "model": args.model
    }



    ####################################
    ### save params and dictionaries ###
    # first folder is dataset
    save_folder = f"{args.model_save_path}/{args.dataset}"
    if args.n_samples is not None:
        save_folder += f"_samples{args.n_samples}"
    # second folder is model
    if args.model == "MLP":
        save_folder += f"/MLP_depth{args.mlp_num_layers}_hidden{args.mlp_hidden_dim}"
    else:
        save_folder += f"/{args.model}"
    # third folder is seed
    save_folder += f"/seed_{args.seed}"
    os.makedirs(save_folder, exist_ok=True)
    
    if args.run_name is not None:
        save_name = f"{args.run_name}"
    else:
        save_name = f"started_{now_string}"

    print(f"Saving to {save_folder}/{save_name}")
    pickle.dump(model_dict, open(f"{save_folder}/{save_name}_params.pickle", "wb"))
    pickle.dump(stats_dict, open(f"{save_folder}/{save_name}_stats.pickle", "wb"))
    with open(f"{save_folder}/{save_name}_args.json", "w") as f:
        json.dump(args_dict, f)