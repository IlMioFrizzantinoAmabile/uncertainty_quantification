import pickle
import json
import flax
from flax import linen as nn
from jax import flatten_util
import jax.numpy as jnp

from src.models import MLP, LeNet
from src.models.resnet import ResNet, ResNetBlock
from src.models.googlenet import GoogleNet
from src.datasets.utils import get_output_dim


def compute_num_params(params):
    vector_params = flatten_util.ravel_pytree(params)[0]
    return vector_params.shape[0]

def compute_norm_params(params):
    vector_params = flatten_util.ravel_pytree(params)[0]
    return jnp.linalg.norm(vector_params).item()

#def has_batchstats(model: flax.linen.Module):
#    return isinstance(model, ResNet) or isinstance(model, GoogleNet)


def load_pretrained_model(
        model_name = "LeNet",
        dataset_name = "MNIST",
        run_name = "example",
        seed = 0,
        n_samples = None,
        save_path = "../models"
    ):

    if n_samples is not None:
        dataset_name += f"_samples{n_samples}"
    #args_file_path = f"{save_path}/{dataset_name}/{model_name}/{run_name}_seed{seed}_args.json" #old
    args_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_args.json"
    args_dict = json.load(open(args_file_path, 'r'))
    #assert dataset_name == args_dict["dataset"]
    #assert model_name == args_dict["model"]

    output_dim = get_output_dim(args_dict["dataset"])
    act_fn = getattr(nn, args_dict["activation_fun"])
    if args_dict["model"] == "MLP":
        model = MLP(
            output_dim = output_dim, 
            num_layers = args_dict["mlp_num_layers"], 
            hidden_dim = args_dict["mlp_hidden_dim"], 
            act_fn = act_fn,
        )
    elif args_dict["model"] == "LeNet":
        model = LeNet(
            output_dim = output_dim,
            act_fn = act_fn,
        )
    elif args_dict["model"] == "GoogleNet":
        model = GoogleNet(
            output_dim = output_dim,
            act_fn = act_fn,
        )
    elif args_dict["model"] == "ResNet":
        model = ResNet(
            output_dim = output_dim,
            c_hidden =(16, 32, 64),
            num_blocks = (3, 3, 3),
            act_fn = act_fn,
            block_class = ResNetBlock,
        )
    elif args_dict["model"] == "ResNet50":
        model = ResNet(
            output_dim = output_dim,
            c_hidden = (32, 64, 128, 256),
            num_blocks = (3, 4, 6, 3),
            act_fn = act_fn,
            block_class = ResNetBlock
        )
    else:
        raise ValueError(f"Model {model_name} unknown")

    #params_file_path = f"{save_path}/{dataset_name}/{model_name}/{run_name}_seed{seed}_params.pickle" #old
    params_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_params.pickle"
    params_dict = pickle.load(open(params_file_path, 'rb'))

    return model, params_dict, args_dict

