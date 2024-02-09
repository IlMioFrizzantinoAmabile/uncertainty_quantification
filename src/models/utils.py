import pickle
import json
import jax
from jax import flatten_util
from src.models import MLP, LeNet
from src.models.resnet import ResNet, ResNetBlock
from src.models.googlenet import GoogleNet
from flax import linen as nn

def load_pretrained_model(
        model_name = "LeNet",
        dataset_name = "MNIST",
        run_name = "example",
        seed = 0,
        save_path = "../models/"
    ):

    args_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_args.json"
    args_dict = json.load(open(args_file_path, 'r'))
    assert run_name == args_dict["dataset"]

    if args_dict["dataset"] in ["Sinusoidal", "UCI"]:
        output_dim = 1 
    elif args_dict["dataset"] == "CelebA":
        output_dim = 40
    elif args_dict["dataset"] == "CIFAR-100":
        output_dim = 100
    else:
        output_dim = 10

    if args_dict["model"] == "MLP":
        model = MLP(
            output_dim=output_dim, 
            num_layers=args_dict["mlp_num_layers"], 
            hidden_dim=args_dict["mlp_hidden_dim"], 
            activation=args_dict["activation_fun"]
        )
    elif args_dict["model"] == "LeNet":
        model = LeNet(
            output_dim=output_dim,
            activation=args_dict["activation_fun"]
        )
    elif args_dict["model"] == "ResNet":
        model = ResNet(
            output_dim = output_dim,
            c_hidden =(16, 32, 64),
            num_blocks = (3, 3, 3),
            act_fn = nn.relu,
            block_class = ResNetBlock 
        )
    elif args_dict["model"] == "GoogleNet":
        model = GoogleNet(
            num_classes = output_dim,
            act_fn = nn.relu
        )
    else:
        raise ValueError(f"Model {model_name} unknown")

    params_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_params.pickle"
    params_dict = pickle.load(open(params_file_path, 'rb'))
    #if not (isinstance(model, ResNet) or isinstance(model, GoogleNet)):
    P = flatten_util.ravel_pytree(params_dict['params'])[0].shape[0]
    print(f"Loaded {args_dict['model']} with {P} parameters")

    return model, params_dict