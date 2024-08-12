import pickle
import os
import argparse
import json
import datetime

from src.datasets import augmented_dataloader_from_string, get_output_dim
from src.models import model_from_string, pretrained_model_from_string
from src.training.trainer import gradient_descent


parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "ImageNet"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--uci_type", type=str, choices=["concrete", "boston", "energy", "kin8nm", "wine", "yacht"], default=None)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet", "GoogleNet", "ConvNeXt", "ConvNeXt_L", "ConvNeXt_XL", "ResNet", "ResNet_NoNorm", "ResNet50", "ResNet50PreAct", "VAN_tiny", "VAN_small", "VAN_base", "VAN_large", "SWIN_tiny", "SWIN_large"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu"], default="tanh", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of layers in the MLP.")

# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="adam")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--decrease_learning_rate", action="store_true", required=False, default=False)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification", "binary_multiclassification"], default="classification")

parser.add_argument("--default_hyperparams", action="store_true", required=False, default=False)

# extra regularizer
parser.add_argument("--regularizer", type=str, choices=["log_determinant_ggn", "log_determinant_ntk"], default=None)
parser.add_argument("--regularizer_hutch_samples", type=int, default=10)
parser.add_argument("--regularizer_prec_prior", type=float, default=1.)
parser.add_argument("--regularizer_prec_lik", type=float, default=1.)
parser.add_argument("--n_warmup_epochs", type=int, default=0)


# storage
parser.add_argument("--run_name", type=str, default=None, help="Fix the save file name. If None it's set to starting time")
parser.add_argument("--run_name_pretrained", type=str, default=None, help="Run name from which to load pretrained parameters. If None parameters are randomly initialized")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--test_every_n_epoch", type=int, default=20, help="Frequency of computing validation stats")

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
        elif args.model == "ConvNeXt" or args.model == "ConvNeXt_L" or args.model == "ConvNeXt_XL":
            args_dict["n_epochs"] = 200
            args_dict["batch_size"] = 128
            args_dict["optimizer"] = "sgd" #"adamw"
            args_dict["learning_rate"] = 0.1 #1e-3
            args_dict["decrease_learning_rate"] = True
            args_dict["momentum"] = 0.9 #None
            args_dict["weight_decay"] = 1e-4
        elif args.model == "ResNet" or args.model == "ResNet_NoNorm":
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
            args_dict["learning_rate"] = 1e-5 #0.0001
            args_dict["decrease_learning_rate"] = True
            args_dict["momentum"] = 0.9
            args_dict["weight_decay"] = 1e-4
            args_dict["activation_fun"] = "relu"
        #elif args.model == "VAN_tiny" or args.model == "VAN_small":
            #args_dict["n_epochs"] = 10 
            #args_dict["batch_size"] = 128 
            #args_dict["optimizer"] = "adam"
            #args_dict["learning_rate"] = 1e-5 
            #args_dict["decrease_learning_rate"] = True
            #args_dict["momentum"] = 0.9
            #args_dict["weight_decay"] = 1e-4

    ###############
    ### dataset ###
    train_loader, valid_loader = augmented_dataloader_from_string(
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
    model = model_from_string(
        args.model, 
        output_dim, 
        activation_fun = args_dict["activation_fun"],
        mlp_num_layers = args_dict["mlp_num_layers"],
        mlp_hidden_dim = args_dict["mlp_hidden_dim"],
    )
    args_dict["output_dim"] = output_dim
    args_dict["opt_hp"] = {
            "lr": args_dict["learning_rate"],
            "momentum": args_dict["momentum"],
            "weight_decay": args_dict["weight_decay"],
        }


    ################
    ### training ###  
    if args.run_name_pretrained is not None:
        _, pretrained_params_dict, _ = pretrained_model_from_string(
            model_name = args.model,
            dataset_name = args.dataset,
            n_samples = args.n_samples,
            run_name = args.run_name_pretrained,
            seed = args.seed,
            save_path = args.model_save_path
        )
    params_dict, stats_dict = gradient_descent(
            model, 
            train_loader, 
            valid_loader, 
            args_dict,
            pretrained_params_dict = None if args.run_name_pretrained is None else pretrained_params_dict
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