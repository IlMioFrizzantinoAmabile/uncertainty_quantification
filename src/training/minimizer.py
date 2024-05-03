import jax
import jax.numpy as jnp
import flax
import torch
import optax
import time
import tqdm

from src.models import compute_num_params, compute_norm_params, has_batchstats
from src.training.loss import calculate_loss_without_batchstats, calculate_loss_with_batchstats


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
    print(f"There are {len(train_loader) } batches every epoch")
    batch = next(iter(train_loader))
    x_init, y_init = jnp.array(batch[0].numpy()), jnp.array(batch[1].numpy())
    print(f"First batch shape: data = {x_init.shape}, target = {y_init.shape}")

    ################################
    # init model and loss function #
    key = jax.random.PRNGKey(args_dict["seed"])
    if not (has_batchstats(model)):
        model_has_batch_stats = False
        params_dict = {
            'params' : model.init(key, x_init),
            'batch_stats' : None,
        }
    else:
        model_has_batch_stats = True
        params_dict =  model.init(key, x_init, train=True)
    print(f"Model has {compute_num_params(params_dict['params'])} parameters")


    if args_dict["likelihood"] in ["classification", "binary_multiclassification"]: #only used for prints
        acc_label = "accuracy"
    elif args_dict["likelihood"] == "regression":
        acc_label = "mse"

    ##################
    # init optimizer #
    optimizer_hparams = args_dict["opt_hp"]
    # learning rate schedule
    if not args_dict["decrease_learning_rate"]:
        lr_schedule = optimizer_hparams.pop('lr')
    else:
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
    opt_state = optimizer.init(params_dict['params'])


    ########################
    # define training step #
    if not model_has_batch_stats:
        @jax.jit
        def train_step(opt_state, params_dict, x, y):
            loss_fn = lambda p: calculate_loss_without_batchstats(
                model, 
                p, 
                x, 
                y, 
                likelihood=args_dict["likelihood"]
            )
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
            loss, (acc_or_sse, ) = ret
            # Update parameters
            param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
            params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
            return opt_state, params_dict, loss, acc_or_sse
    else:
        @jax.jit
        def train_step(opt_state, params_dict, x, y):
            loss_fn = lambda p: calculate_loss_with_batchstats(
                model, 
                p, 
                params_dict['batch_stats'], 
                x, 
                y, 
                train=True, 
                likelihood=args_dict["likelihood"]
            )
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
            loss, (acc_or_sse, new_model_state) = ret
            # Update parameters and batch statistics
            param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
            params_dict = {
                'params' : optax.apply_updates(params_dict['params'], param_updates),
                'batch_stats' : new_model_state['batch_stats']
            }
            return opt_state, params_dict, loss, acc_or_sse

    #########################
    # start training epochs #
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
        loss = 0.
        acc_or_sse = 0. if args_dict["likelihood"] != "binary_multiclassification" else jnp.zeros((y_init.shape[1], ))
        start_time = time.time()
        train_loader_bar = tqdm.tqdm(train_loader) if args_dict["verbose"] else train_loader
        for batch in train_loader_bar:
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y)

            loss += batch_loss.item()
            acc_or_sse += batch_acc_or_sse/X.shape[0]

            if args_dict["verbose"]:
                formatted_batch_acc = f"{batch_acc_or_sse/X.shape[0]:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{a:.2f}" for a in batch_acc_or_sse/X.shape[0]]
                train_loader_bar.set_description(f"Epoch {epoch}/{args_dict['n_epochs']}, batch loss = {batch_loss.item():.2f}, {acc_label} = {formatted_batch_acc}")
    
        acc_or_sse /= len(train_loader)
        loss /= len(train_loader)
        params_norm = compute_norm_params(params_dict['params'])
        batch_sats_norm = compute_norm_params(params_dict['batch_stats'])
        acc_or_sse_formatted = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{a:.3f}" for a in acc_or_sse]
        print(f"epoch={epoch} averages - loss={loss:.2f}, params norm={params_norm:.2f}, batch_stats norm={batch_sats_norm:.2f}, {acc_label}={acc_or_sse_formatted}, time={time.time() - start_time:.3f}s")
        epoch_stats_dict["loss"].append(loss)
        epoch_stats_dict["acc_or_mse"].append(acc_or_sse)
        epoch_stats_dict["params_norm"].append(params_norm)


        if epoch % args_dict["test_every_n_epoch"] != 0 and epoch != args_dict["n_epochs"]:
            continue

        def get_precise_stats(loader):
            loss = 0.
            acc_or_sse = 0. if args_dict["likelihood"] != "binary_multiclassification" else jnp.zeros((y_init.shape[1], ))
            start_time = time.time()
            for batch in loader:
                X = jnp.array(batch[0].numpy())
                Y = jnp.array(batch[1].numpy())
                if model_has_batch_stats:
                    batch_loss, (batch_acc_or_sse, _) = calculate_loss_with_batchstats(
                        model, 
                        params_dict['params'], 
                        params_dict['batch_stats'], 
                        X, 
                        Y, 
                        train=False, 
                        likelihood=args_dict["likelihood"]
                    )
                else:
                    batch_loss, (batch_acc_or_sse, ) = calculate_loss_without_batchstats(
                        model, 
                        params_dict['params'], 
                        X, 
                        Y, 
                        likelihood=args_dict["likelihood"],
                    )
                loss += batch_loss.item()
                acc_or_sse += batch_acc_or_sse/X.shape[0]
            acc_or_sse /= len(loader)
            loss /= len(loader)
            return loss, acc_or_sse, time.time() - start_time

        loss, acc_or_sse, duration = get_precise_stats(train_loader)
        acc_or_sse_formatted = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{a:.3f}" for a in acc_or_sse]
        print(f"Train stats\t - loss={loss:.3f}, {acc_label}={acc_or_sse_formatted}, time={duration:.3f}s")
        train_stats_dict["loss"].append(loss)
        train_stats_dict["acc_or_mse"].append(acc_or_sse)

        loss, acc_or_sse, duration = get_precise_stats(valid_loader)
        acc_or_sse_formatted = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{a:.3f}" for a in acc_or_sse]
        print(f"Validation stats - loss={loss:.3f}, {acc_label}={acc_or_sse_formatted} time={duration:.3f}s")
        valid_stats_dict["loss"].append(loss)
        valid_stats_dict["acc_or_mse"].append(acc_or_sse)

    epoch_stats_dict = {'epoch_'+k : v for k,v in epoch_stats_dict.items()}    
    train_stats_dict = {'train_'+k : v for k,v in train_stats_dict.items()}    
    valid_stats_dict = {'valid_'+k : v for k,v in valid_stats_dict.items()}    
    stats_dict = {**train_stats_dict, **valid_stats_dict, **epoch_stats_dict}

    return params_dict, stats_dict