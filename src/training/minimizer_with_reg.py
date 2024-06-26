import jax
import jax.numpy as jnp
import flax
import torch
import optax
import time

from src.models import compute_num_params, compute_norm_params, has_batchstats
from src.training.loss import calculate_loss_without_batchstats, calculate_loss_with_batchstats
from src.training.regularizer import log_determinant_ntk, log_determinant_ggn


def maximum_a_posteriori_with_reg(
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


    #################################
    # extra regularization function #
    if args_dict["regularizer"] is None:
        if not model_has_batch_stats:
            regg_fn = lambda n, x, p: 0.
        else:
            regg_fn = lambda n, x, p, bs: 0.
    else:
        if args_dict["regularizer"] == "log_determinant_ggn":
            log_determinant_fn = log_determinant_ggn
            log_determinant_dim = compute_num_params(params_dict['params'])
        elif args_dict["regularizer"] == "log_determinant_ntk":
            log_determinant_fn = log_determinant_ntk
            log_determinant_dim = args_dict["output_dim"] * args_dict["batch_size"]

        if not model_has_batch_stats:
            regg_fn = lambda n, x, p: log_determinant_fn(
                model, 
                {'params': p}, 
                x, 
                log_determinant_dim,
                prior_precision = args_dict["regularizer_prec_prior"],
                lik_precision = args_dict["regularizer_prec_lik"],
                likelihood = args_dict["likelihood"],
                sequential = False,
                n_samples = args_dict["regularizer_hutch_samples"],
                key = jax.random.PRNGKey(n),
                estimator = "Rademacher"
            ) / log_determinant_dim
        else:
            regg_fn = lambda n, x, p, bs: log_determinant_fn(
                model, 
                {'params': p, 'batch_stats': bs}, 
                x, 
                log_determinant_dim,
                prior_precision = args_dict["regularizer_prec_prior"],
                lik_precision = args_dict["regularizer_prec_lik"],
                likelihood = args_dict["likelihood"],
                sequential = False,
                n_samples = args_dict["regularizer_hutch_samples"],
                key = jax.random.PRNGKey(n),
                estimator = "Rademacher"
            ) / log_determinant_dim
    reg_fn = jax.jit(regg_fn)


    ########################
    # define training step #
    if not model_has_batch_stats:
        @jax.jit
        def train_step(n_epoch, opt_state, params_dict, x, y, reg_scale=1.):
            losss_fn = lambda p: calculate_loss_without_batchstats(
                model, 
                p, 
                x, 
                y, 
                likelihood=args_dict["likelihood"]
            )
            def loss_fn(p):
                loss, (acc, ) = losss_fn(p)
                reg = reg_scale * reg_fn(n_epoch, x, p)
                #reg = reg_fn(n_epoch, x, p)
                #print(loss, reg)
                return loss+reg, (acc, )
            # Get loss, gradients for loss, and other outputs of loss function
            #loss, (acc_or_sse, ) = loss_fn(params_dict['params'])
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
            loss, (acc_or_sse, ) = ret
            # Update parameters
            param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
            params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
            return opt_state, params_dict, loss, acc_or_sse
    else:
        @jax.jit
        def train_step(n_epoch, opt_state, params_dict, x, y, reg_scale=1.):
            losss_fn = lambda p: calculate_loss_with_batchstats(
                model, 
                p, 
                params_dict['batch_stats'], 
                x, 
                y, 
                train=True, 
                likelihood=args_dict["likelihood"]
            )
            def loss_fn(p):
                loss, (acc, new_model_state) = losss_fn(p)
                reg = reg_scale * reg_fn(n_epoch, x, p, params_dict['batch_stats'])
                #reg = reg_fn(n_epoch, x, p, params_dict['batch_stats'])
                return loss+reg, (acc, new_model_state)
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
        reg_scale = 0. if epoch<args_dict["n_warmup_epochs"] else 1.
        loss, acc_or_sse = 0., 0.
        start_time = time.time()
        for batch in train_loader:
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(epoch, opt_state, params_dict, X, Y, reg_scale=reg_scale)

            loss += batch_loss.item()
            acc_or_sse += batch_acc_or_sse/X.shape[0]
    
        acc_or_sse /= len(train_loader)
        loss /= len(train_loader)
        params_norm = compute_norm_params(params_dict['params'])
        batch_sats_norm = compute_norm_params(params_dict['batch_stats'])
        if args_dict["likelihood"] in ["classification", "binary_multiclassification"]:
            print(f"epoch={epoch} averages - loss={loss:.2f}, params norm={params_norm:.2f}, batch_stats norm={batch_sats_norm:.2f}, accuracy={acc_or_sse:.3f}, time={time.time() - start_time:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"epoch={epoch} averages - loss={loss:.2f}, params norm={params_norm:.2f}, batch_stats norm={batch_sats_norm:.2f}, mse={acc_or_sse:.3f}, time={time.time() - start_time:.3f}s")
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
        if args_dict["likelihood"] in ["classification", "binary_multiclassification"]:
            print(f"Train stats\t - loss={loss:.3f}, accuracy={acc_or_sse:.3f}, time={duration:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"Train stats\t - loss={loss:.3f}, mse={acc_or_sse:.3f}, time={duration:.3f}s")
        train_stats_dict["loss"].append(loss)
        train_stats_dict["acc_or_mse"].append(acc_or_sse)

        loss, acc_or_sse, duration = get_precise_stats(valid_loader)
        if args_dict["likelihood"] in ["classification", "binary_multiclassification"]:
            print(f"Validation stats - loss={loss:.3f} accuracy={acc_or_sse:.3f}, time={duration:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"Validation stats - loss={loss:.3f}, mse={acc_or_sse:.3f} time={duration:.3f}s")
        valid_stats_dict["loss"].append(loss)
        valid_stats_dict["acc_or_mse"].append(acc_or_sse)

    epoch_stats_dict = {'epoch_'+k : v for k,v in epoch_stats_dict.items()}    
    train_stats_dict = {'train_'+k : v for k,v in train_stats_dict.items()}    
    valid_stats_dict = {'valid_'+k : v for k,v in valid_stats_dict.items()}    
    stats_dict = {**train_stats_dict, **valid_stats_dict, **epoch_stats_dict}

    return params_dict, stats_dict