import pickle
import json
import dataclasses
from typing import Callable 
import jax
import flax
from src.models import MLP, LeNet, GoogleNet, ConvNeXt, ResNet, ResNetBlock, PreActResNetBlock, VAN, SwinTransformer


@dataclasses.dataclass
class Model:
    init : Callable # init(x, params)
    apply_train : Callable  # apply(x, params)
    apply_test : Callable  # apply(x, params)
    has_batch_stats: bool
    has_dropout: bool
    has_attentionmask: bool

def wrap_model(model) -> Model:

    def init(key, x):
        params_dict = {
            'params' : model.init(key, x),
            'batch_stats' : None,
        }
        return params_dict

    def apply_train(params, x):
        return model.apply(params, x)
    
    def apply_test(params, x):
        return model.apply(params, x)
    
    return Model(init=init, apply_train=apply_train, apply_test=apply_test, has_batch_stats=False, has_dropout=False, has_attentionmask=False)

def wrap_model_with_dropout(model) -> Model:

    def init(key, x):
        params_dict = model.init({"params": key}, x, deterministic=True)
        return params_dict

    def apply_train(params, x, key_dropout):
        key_dropout, key_dropout2 = jax.random.split(key_dropout, 2)
        return model.apply(
                params,
                x,
                deterministic=False,
                rngs={'drop_path': key_dropout, 'dropout': key_dropout2})
    
    def apply_test(params, x):
        return model.apply(
                params,
                x,
                deterministic=True)
    
    return Model(init=init, apply_train=apply_train, apply_test=apply_test, has_batch_stats=False, has_dropout=True, has_attentionmask=False)

def wrap_model_with_batchstats(model) -> Model:

    def init(key, x):
        params_dict = model.init(key, x, train=True)
        return params_dict

    def apply_train(params, batch_stats, x):
        return model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                train=True,
                mutable=['batch_stats'])
    
    def apply_test(params, batch_stats, x):
        return model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                train=False,
                mutable=False)
    
    return Model(init=init, apply_train=apply_train, apply_test=apply_test, has_batch_stats=True, has_dropout=False, has_attentionmask=False)

def wrap_model_with_batchstats_dropout(model) -> Model:

    def init(key, x):
        #key, drop = jax.random.split(key, 2)
        #params_dict = model.init({"params": key, "drop_path": drop}, x)
        params_dict = model.init({"params": key}, x, deterministic=True)
        return params_dict

    def apply_train(params, batch_stats, x, key_dropout):
        key_dropout, key_dropout2 = jax.random.split(key_dropout, 2)
        return model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                deterministic=False,
                #rngs={'drop_path': key_dropout},
                rngs={'drop_path': key_dropout, 'dropout': key_dropout2},
                mutable=['batch_stats'])
    
    def apply_test(params, batch_stats, x):
        return model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                deterministic=True,
                mutable=False)
    
    return Model(init=init, apply_train=apply_train, apply_test=apply_test, has_batch_stats=True, has_dropout=True, has_attentionmask=False)


def wrap_model_with_attentionmask(model) -> Model:

    def init(key, x):
        rng1, rng2, rng3 = jax.random.split(key, 3)
        params_dict = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )
        return params_dict

    def apply_train(params, attention_mask, relative_position_index, x, key_dropout):
        key_dropout, key_dropout2 = jax.random.split(key_dropout, 2)
        return model.apply(
                {
                    'params': params, 
                    'attention_mask': attention_mask, 
                    'relative_position_index': relative_position_index
                },
                x,
                deterministic=False,
                rngs={'drop_path': key_dropout, 'dropout': key_dropout2},
                mutable=['attention_mask', 'relative_position_index'])
    
    def apply_test(params, attention_mask, relative_position_index, x):
        return model.apply(
                {
                    'params': params, 
                    'attention_mask': attention_mask, 
                    'relative_position_index': relative_position_index
                },
                x,
                deterministic=True,
                mutable=False)
    
    return Model(init=init, apply_train=apply_train, apply_test=apply_test, has_batch_stats=False, has_dropout=True, has_attentionmask=True)


def model_from_string(
        model_name: str, 
        output_dim: int, 
        activation_fun: str = "relu",
        mlp_num_layers: int = 1,
        mlp_hidden_dim: int = 20,
    ):
    
    act_fn = getattr(flax.linen, activation_fun)

    if model_name == "MLP":
        model = MLP(
            output_dim = output_dim, 
            num_layers = mlp_num_layers,
            hidden_dim = mlp_hidden_dim, 
            act_fn = act_fn
        )
        wrapped_model = wrap_model(model)
    elif model_name == "LeNet":
        model = LeNet(
            output_dim = output_dim, 
            act_fn = act_fn
        )
        wrapped_model = wrap_model(model)
    elif model_name == "GoogleNet":
        model = GoogleNet(
            output_dim = output_dim,
            act_fn = act_fn
        )
        wrapped_model = wrap_model_with_batchstats(model)
    elif model_name == "ConvNeXt":
        model = ConvNeXt(
            depths = (3, 3, 9, 3),
            dims = (16, 32, 64, 128),
            drop_path = 0.0,
            attach_head = True,
            num_classes = output_dim,
            deterministic = True
        )
        wrapped_model = wrap_model(model)
    elif model_name == "ConvNeXt_L":
        model = ConvNeXt(
            depths = (3, 3, 9, 3),
            dims = (32, 64, 128, 256),
            drop_path = 0.0,
            attach_head = True,
            num_classes = output_dim,
            deterministic = True
        )
        wrapped_model = wrap_model(model)
    elif model_name == "ConvNeXt_XL":
        model = ConvNeXt(
            depths = (3, 3, 27, 3),
            dims = (128, 256, 512, 1024),
            drop_path = 0.0,
            attach_head = True,
            num_classes = output_dim,
            deterministic = True
        )
        wrapped_model = wrap_model(model)
    elif model_name == "ResNet":
        model = ResNet(
            output_dim = output_dim,
            c_hidden = (16, 32, 64),
            num_blocks = (3, 3, 3),
            act_fn = act_fn,
            block_class = ResNetBlock
        )
        wrapped_model = wrap_model_with_batchstats(model)
    elif model_name == "ResNet50":
        model = ResNet(
            output_dim = output_dim,
            c_hidden = (32, 64, 128, 256),
            num_blocks = (3, 4, 6, 3),
            act_fn = act_fn,
            block_class = ResNetBlock
        )
        wrapped_model = wrap_model_with_batchstats(model)
    elif model_name == "ResNet50PreAct":
        model = ResNet(
            output_dim = output_dim,
            c_hidden = (32, 64, 128, 256),
            num_blocks = (3, 4, 6, 3),
            act_fn = act_fn,
            block_class = PreActResNetBlock
        )
        wrapped_model = wrap_model_with_batchstats(model)
    elif model_name == "VAN_tiny":
        model = VAN(
            embed_dims=(32, 64, 160, 256),
            mlp_ratios=(8, 8, 4, 4),
            depths=(3, 3, 5, 2),
            dropout=0., #0.4,
            drop_path=0.3, #0.5,
            attach_head=True,
            deterministic=False,
            num_classes=output_dim,
        )
        wrapped_model = wrap_model_with_batchstats_dropout(model)
    elif model_name == "VAN_small":
        model = VAN(
            embed_dims = (64, 128, 320, 512),
            mlp_ratios = (8, 8, 4, 4),
            depths = (2, 2, 4, 2),
            attach_head = True,
            num_classes = output_dim,
        )
        wrapped_model = wrap_model_with_batchstats_dropout(model)
    elif model_name == "VAN_base":
        model = VAN(
            embed_dims=(64, 128, 320, 512),
            mlp_ratios=(8, 8, 4, 4),
            depths=(3, 3, 12, 3),
            attach_head = True,
            num_classes = output_dim,
        )
        wrapped_model = wrap_model_with_batchstats_dropout(model)
    elif model_name == "VAN_large":
        model = VAN(
            embed_dims = (64, 128, 320, 512),
            mlp_ratios = (8, 8, 4, 4),
            depths = (3, 5, 27, 3),
            dropout=0.,
            drop_path=0.1,
            attach_head=True,
            deterministic=False,
            num_classes = output_dim,
        )
        wrapped_model = wrap_model_with_batchstats_dropout(model)
    elif model_name == "SWIN_tiny":
        model = SwinTransformer(
            patch_size=4,
            emb_dim=96,
            depths=(1,1), #(2, 2, 6, 2),
            num_heads=(3,3), #(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4,
            use_att_bias=True,
            dropout=0.0,
            att_dropout=0.0,
            drop_path=0.1,
            use_abs_pos_emb=False,
            attach_head=True,
            num_classes=output_dim,
        )
        wrapped_model = wrap_model_with_attentionmask(model)
    elif model_name == "SWIN_large":
        model = SwinTransformer(
            patch_size=4,
            emb_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            window_size=7,
            mlp_ratio=4,
            use_att_bias=True,
            dropout=0.0,
            att_dropout=0.0,
            drop_path=0.1,
            use_abs_pos_emb=False,
            attach_head=True,
            num_classes=output_dim,
        )
        wrapped_model = wrap_model_with_attentionmask(model)
    else:
        raise ValueError(f"Model {model_name} is not implemented (yet)")

    return wrapped_model



def pretrained_model_from_string(
        model_name = "LeNet",
        dataset_name = "MNIST",
        run_name = "example",
        seed = 0,
        n_samples = None,
        save_path = "../models"
    ):

    if n_samples is not None:
        dataset_name += f"_samples{n_samples}"
    args_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_args.json"
    args_dict = json.load(open(args_file_path, 'r'))

    extra_args = {
        "activation_fun" : args_dict["activation_fun"],
        "mlp_num_layers" : args_dict["mlp_num_layers"],
        "mlp_hidden_dim" : args_dict["mlp_hidden_dim"],
    }
   
    model = model_from_string(args_dict["model"], args_dict["output_dim"], **extra_args)

    params_file_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_params.pickle"
    params_dict = pickle.load(open(params_file_path, 'rb'))
    params_dict.pop("model")

    return model, params_dict, args_dict

