import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import torch
import optax
import time
from typing import Any

from src.models.resnet import ResNet
from src.models.googlenet import GoogleNet
from src.training.loss import calculate_loss_without_batchstats, calculate_loss_with_batchstats, compute_num_params, compute_norm_params


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any

def maximum_a_posteriori(
    model: flax.linen.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    args_dict: dict,
):
    """
    Maximize the posterior for a given model and dataset.
    :param model: initialized model to use for training
    :param train_loader: train dataloader (torch.utils.data.DataLoader)
    :param valid_loader: test dataloader (torch.utils.data.DataLoader)
    :param key: random.PRNGKey for jax modules
    :param args_dict: dictionary of arguments for training passed from the command line
    :return: params
    """
    #################
    # observe datas #
    N = len(train_loader.dataset) 
    batch = next(iter(train_loader))
    x_init, y_init = jnp.array(batch[0].numpy()), jnp.array(batch[1].numpy())
    print(f"First batch shape: data = {x_init.shape}, target = {y_init.shape}")

    ################################
    # init model and loss function #
    key = jax.random.PRNGKey(args_dict["seed"])
    if not (isinstance(model, ResNet) or isinstance(model, GoogleNet)):
        model_has_batch_stats = False
        params_dict = {
            'params' : model.init(key, x_init),
            'batch_stats' : None,
        }
        @jax.jit
        def train_step(state, x, y):
            loss_fn = lambda params: calculate_loss_without_batchstats(
                model, 
                params, 
                x, 
                y, 
                likelihood=args_dict["likelihood"]
            )
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, (acc_or_sse, ) = ret
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads)
            return state, loss, acc_or_sse
    else:
        model_has_batch_stats = True
        params_dict =  model.init(key, x_init, train=True)
        @jax.jit
        def train_step(state, x, y):
            loss_fn = lambda params: calculate_loss_with_batchstats(
                model, 
                params, 
                state.batch_stats, 
                x, 
                y, 
                train=True, 
                likelihood=args_dict["likelihood"]
            )
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, (acc_or_sse, new_model_state) = ret
            # Update parameters
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, loss, acc_or_sse
    print(f"Model has {compute_num_params(params_dict['params'])} parameters")

    ##################
    # init optimizer #
    optimizer_hparams = args_dict["opt_hp"]
    # learning rate schedule
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=optimizer_hparams.pop('lr'),
        boundaries_and_scales={
            int(len(train_loader)*args_dict["n_epochs"]*0.6): 0.1,
            int(len(train_loader)*args_dict["n_epochs"]*0.85): 0.1
        }
    )
    # clip gradient
    transf = [optax.clip(1.0)]
    # add weight decay if requested
    #if args_dict["optimizer"] == 'sgd' and optimizer_hparams['weight_decay'] is not None:  
    if optimizer_hparams['momentum'] is None:
        optimizer_hparams.pop('momentum')
    if optimizer_hparams['weight_decay'] is None:
        optimizer_hparams.pop('weight_decay')
    elif args_dict["optimizer"] in ['sgd', 'adam']:
        # weight decay is integrated in adamw
        transf.append(optax.add_decayed_weights(optimizer_hparams.pop('weight_decay')))
    # define optimizer
    opt_class = getattr(optax, args_dict["optimizer"])
    optimizer = optax.chain(
        *transf,
        opt_class(lr_schedule, **optimizer_hparams)
    )

    #########################
    # start training epochs #
    state = TrainState.create(
        apply_fn = model.apply,
        params = params_dict['params'],
        batch_stats = params_dict['batch_stats'],
        tx = optimizer
    )
    epoch_stats_dict = { # computed while parameters change 
        "loss": [],
        "acc_or_mse": [],
        "params_norm" : [] }
    train_stats_dict = { # computed with fixed parameters
        "loss": [],
        "acc_or_mse": [] }
    valid_stats_dict = { # computed with fixed parameters
        "loss": [],
        "acc_or_mse": [] }
    print("Starting training...")
    for epoch in range(1, args_dict["n_epochs"] + 1):
        loss, acc_or_sse = 0., 0.
        n_batches = 0
        start_time = time.time()
        for batch in train_loader:
            n_batches += 1
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            state, batch_loss, batch_acc_or_sse = train_step(state, X, Y)

            loss += batch_loss.item()
            acc_or_sse += batch_acc_or_sse
    
        acc_or_sse /= len(train_loader.dataset)
        params_norm = compute_norm_params(state.params)
        if args_dict["likelihood"] == "classification":
            print(f"epoch={epoch} averages - loss={loss:.2f}, params norm={params_norm:.2f}, accuracy={acc_or_sse:.2f}, time={time.time() - start_time:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"epoch={epoch} averages - loss={loss:.2f}, params norm={params_norm:.2f}, mse={acc_or_sse:.2f}, time={time.time() - start_time:.3f}s")
        epoch_stats_dict["loss"].append(loss)
        epoch_stats_dict["acc_or_mse"].append(acc_or_sse)
        epoch_stats_dict["params_norm"].append(params_norm)

        if epoch % args_dict["test_every_n_epoch"] != 0 and epoch != args_dict["n_epochs"]:
            continue

        def get_precise_stats(loader):
            loss, acc_or_sse = 0., 0.
            start_time = time.time()
            for batch in loader:
                X = jnp.array(batch[0].numpy())
                Y = jnp.array(batch[1].numpy())
                if model_has_batch_stats:
                    batch_loss, (batch_acc_or_sse, _) = calculate_loss_with_batchstats(
                        model, 
                        state.params, 
                        state.batch_stats, 
                        X, 
                        Y, 
                        train=False, 
                        likelihood=args_dict["likelihood"]
                    )
                else:
                    batch_loss, (batch_acc_or_sse, ) = calculate_loss_without_batchstats(
                        model, 
                        state.params, 
                        X, 
                        Y, 
                        likelihood=args_dict["likelihood"]
                    )
                loss += batch_loss.item()
                acc_or_sse += batch_acc_or_sse
            acc_or_sse /= len(loader.dataset)
            return loss, acc_or_sse, time.time() - start_time

        loss, acc_or_sse, duration = get_precise_stats(train_loader)
        if args_dict["likelihood"] == "classification":
            print(f"Train stats\t - loss={loss:.3f}, accuracy={acc_or_sse:.2f}, time={duration:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"Train stats\t - loss={loss:.3f}, mse={acc_or_sse:.2f}, time={duration:.3f}s")
        train_stats_dict["loss"].append(loss)
        train_stats_dict["acc_or_mse"].append(acc_or_sse)

        loss, accuracy, duration = get_precise_stats(valid_loader)
        if args_dict["likelihood"] == "classification":
            print(f"Validation stats - loss={loss:.3f} accuracy={acc_or_sse:.2f}, time={duration:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"Validation stats - loss={loss:.3f}, mse={acc_or_sse:.2f} time={duration:.3f}s")
        valid_stats_dict["loss"].append(loss)
        valid_stats_dict["acc_or_mse"].append(acc_or_sse)

    epoch_stats_dict = {'epoch_'+k : v for k,v in epoch_stats_dict.items()}    
    train_stats_dict = {'train_'+k : v for k,v in train_stats_dict.items()}    
    valid_stats_dict = {'valid_'+k : v for k,v in valid_stats_dict.items()}    
    stats_dict = {**train_stats_dict, **valid_stats_dict, **epoch_stats_dict}

    return state.params, state.batch_stats, stats_dict