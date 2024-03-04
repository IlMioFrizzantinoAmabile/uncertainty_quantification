import pickle
import os
import argparse
import json
import datetime
from flax import linen as nn

from src.datasets import get_train_loaders, get_output_dim
from src.models import MLP, LeNet, GoogleNet, ResNet, ResNetBlock, PreActResNetBlock

from src.training.minimizer import maximum_a_posteriori
from src.training.minimizer_with_reg import maximum_a_posteriori_with_reg


parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--uci_type", type=str, choices=["concrete", "boston", "energy", "kin8nm", "wine", "yacht"], default=None)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet", "GoogleNet", "ResNet", "ResNet50", "ResNet50PreAct"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu"], default="tanh", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of layers in the MLP.")

# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "rmsprop"], default="adam")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--decrease_learning_rate", action="store_true", required=False, default=False)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification", "binary_multiclassification"], default="classification")

parser.add_argument("--default_hyperparams", action="store_true", required=False, default=False)

# extra regularizer
parser.add_argument("--regularizer", type=str, choices=["log_determinant_ggn", "log_determinant_ntk"], default=None)
parser.add_argument("--regularizer_hutch_samples", type=int, default=10)

# storage
parser.add_argument("--run_name", default=None, help="Fix the save file name. If None it's set to starting time")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--test_every_n_epoch", type=int, default=20, help="Frequency of coputing validation stats")

# print more stuff
parser.add_argument("--verbose", action="store_true", required=False, default=False)



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # some reasonable hyperparameters
    if args.default_hyperparams:
        if args.model in ["MLP", "LeNet"]:
            args_dict["n_epochs"] = 50
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "adam"
            args_dict["learning_rate"] = 1e-3
            args_dict["momentum"] = None
            args_dict["weight_decay"] = None
            args_dict["activation_fun"] = "tanh"
        elif args.model == "GoogleNet":
            args_dict["n_epochs"] = 200
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "adamw"
            args_dict["learning_rate"] = 1e-3
            args_dict["decrease_learning_rate"] = True
            args_dict["momentum"] = None
            args_dict["weight_decay"] = 1e-4
            args_dict["activation_fun"] = "relu"
        elif args.model == "ResNet":
            args_dict["n_epochs"] = 200
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "sgd"
            args_dict["learning_rate"] = 0.1
            args_dict["decrease_learning_rate"] = True
            args_dict["momentum"] = 0.9
            args_dict["weight_decay"] = 1e-4
            args_dict["activation_fun"] = "relu"
        elif args.model == "ResNet50":
            args_dict["n_epochs"] = 2
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "sgd"
            args_dict["learning_rate"] = 0.0001
            args_dict["decrease_learning_rate"] = True
            args_dict["momentum"] = 0.9
            args_dict["weight_decay"] = 1e-4
            args_dict["activation_fun"] = "relu"

    ###############
    ### dataset ###
    train_loader, valid_loader = get_train_loaders(
        args.dataset,
        n_samples = args.n_samples,
        batch_size = args.batch_size,
        shuffle = True,
        seed = args.seed,
        download = True,
        data_path = args.data_path
    )
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}")


    #############
    ### model ###
    output_dim = get_output_dim(args.dataset)
    args_dict["output_dim"] = output_dim
    act_fn = getattr(nn, args_dict["activation_fun"])

    have_batch_stats = False
    if args.model == "MLP":
        model = MLP(
            output_dim = output_dim, 
            num_layers = args.mlp_num_layers,
            hidden_dim = args.mlp_hidden_dim, 
            act_fn = act_fn
        )
    elif args.model == "LeNet":
        model = LeNet(
            output_dim = output_dim, 
            act_fn = act_fn
        )
    elif args.model == "GoogleNet":
        model = GoogleNet(
            output_dim = output_dim,
            act_fn = act_fn
        )
    elif args.model == "ResNet":
        model = ResNet(
            output_dim = output_dim,
            c_hidden =(16, 32, 64),
            num_blocks = (3, 3, 3),
            act_fn = act_fn,
            block_class = ResNetBlock
        )
    elif args.model == "ResNet50":
        model = ResNet(
            output_dim = output_dim,
            c_hidden = (32, 64, 128, 256), #(16, 32, 64, 128, 256, 512),
            num_blocks = (3, 4, 6, 3),
            act_fn = act_fn,
            block_class = ResNetBlock
        )
    elif args.model == "ResNet50PreAct":
        model = ResNet(
            output_dim = output_dim,
            c_hidden = (32, 64, 128, 256), #(16, 32, 64, 128, 256, 512),
            num_blocks = (3, 4, 6, 3),
            act_fn = act_fn,
            block_class = PreActResNetBlock
        )
    else:
        raise ValueError(f"Model {args.model} is not implemented")
    args_dict["opt_hp"] = {
            "lr": args_dict["learning_rate"],
            "momentum": args_dict["momentum"],
            "weight_decay": args_dict["weight_decay"],
        }


    ################
    ### training ###  
    if args.regularizer is None:
        params_dict, stats_dict = maximum_a_posteriori(
                model, train_loader, valid_loader, args_dict
            )
    else:
        params_dict, stats_dict = maximum_a_posteriori_with_reg(
                model, train_loader, valid_loader, args_dict
            )
    model_dict = {"model": args.model, **params_dict}



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