import jax
import jax.numpy as jnp
import torch
import optax
import time
import tqdm
from functools import partial
from typing import Union
from jax import flatten_util

from matfree import funm, stochtrace
from src.autodiff.ntk import get_ntk_vector_product
from src.autodiff.ggn import get_ggn_vector_product
from src.lanczos.high_memory import high_memory_lanczos

from src.models import compute_num_params, compute_norm_params, Model
from src.training.losses import get_loss_function

from src.autodiff.projection import get_projection_vector_product
from src.datasets.utils import get_subset_loader


from src.datasets import augmented_dataloader_from_string


def gradient_descent_fancy(
    model: Model,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    args_dict: dict,
    pretrained_params_dict: dict = None,
):
    train_loader, valid_loader = augmented_dataloader_from_string(
        "Sinusoidal",
        n_samples = None,
        batch_size = 100,
        shuffle = False,
        seed = 1,
    )

    """
    Mimimize the loss for a given model and dataset.
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

    ##############
    # init model #
    key = jax.random.PRNGKey(args_dict["seed"])
    if model.has_dropout:
        key, key_dropout = jax.random.split(key, 2)
    if pretrained_params_dict is None:
        params_dict = model.init(key, x_init)
    else:
        params_dict = pretrained_params_dict
    print(f"Model has {compute_num_params(params_dict['params'])} parameters")
    P = compute_num_params(params_dict['params'])

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

    #ntk_matvec = get_ntk_vector_product(params_dict, model, x_init, likelihood_type="regression")



    ntk_matvec = get_ntk_vector_product(params_dict, model, x_init, likelihood_type="regression")
    matvec = lambda v: 1. * v + 1. * ntk_matvec(v)
    identity = jnp.eye(100)
    ntk = jax.vmap(matvec)(identity)
    resss = model.apply_test(params_dict['params'],x_init).reshape(-1) - y_init.reshape(-1)
    log_det = jax.numpy.linalg.slogdet(ntk)[1]
    ntk_inv_resss = jax.scipy.linalg.solve(ntk, resss)
    true_marginal_log_lik = - 0.5 * log_det - 0.5 * resss @ ntk_inv_resss
    print(f"log det: {log_det} - residual: {resss @ ntk_inv_resss}")
    print(f"THE TRUE ONE: {true_marginal_log_lik}")





    ###################
    # kernel manifold #

    def get_distance_from_kernel_manifold(model):
        #@jax.jit
        def distance_from_kernel_manifold(params, x, preds):
            #residual = model.apply_test(params,x).reshape(-1) - preds.reshape(-1)
            residual = model.apply_test(params,x) - preds
            return jnp.sum( residual**2 )
        return distance_from_kernel_manifold
    distance_from_kernel_manifold = get_distance_from_kernel_manifold(model)
    def get_projection_on_kernel_manifold_step(x, preds):
        @jax.jit
        def projection_on_kernel_manifold_step(params_dict):
            distance = lambda p: distance_from_kernel_manifold(p, x, preds)
            ret, grads = jax.value_and_grad(distance)(params_dict['params'])
            params_vector, unflatten_fun = flatten_util.ravel_pytree(params_dict['params'])
            grads_vector = flatten_util.ravel_pytree(grads)[0]
            params_vector = params_vector - 0.0001 * grads_vector
            params_dict['params'] = unflatten_fun(params_vector)
            return params_dict
        return projection_on_kernel_manifold_step
    def get_projection_on_kernel_manifold(x, preds):
        projection_on_kernel_manifold_step = get_projection_on_kernel_manifold_step(x, preds)
        step = lambda i, params_dict: projection_on_kernel_manifold_step(params_dict)
        @jax.jit
        def projection_on_kernel_manifold(params_dict):
            return jax.lax.fori_loop(0, 10000, step, params_dict)
        return projection_on_kernel_manifold


    #############
    # init loss #
    def get_marginal_loss(model):
        @partial(jax.jit, static_argnames=['rho', 'alpha'])
        def loss_function_test(params, x, y, rho=1., alpha=1.0):
            
            #NTK = 1./rho + 1./alpha * J * J.t
            #marginal_log_lik = - 0.5 * logdet(NTK) - 0.5 * (model(params,x) - y) * NTK^(-1) * (model(params,x) - y)
            residual = model.apply_test(params,x).reshape(-1) - y.reshape(-1)
            mse = jnp.sqrt(jnp.sum( residual**2 ))

            if True:
                params_dict = {"params": params}
                ntk_matvec = get_ntk_vector_product(params_dict, model, x, likelihood_type="regression")
                matvec = lambda v: 1./rho * v + 1./alpha * ntk_matvec(v)
                y_flatten = y.reshape(-1)
                #print(matvec(y_flatten).shape)

                #good_shit = jnp.linalg.norm(matvec(y_flatten))

                order = 6 #20 # goood
                num = 100 #goood
                inv_iterations = 6 #30 #goood

                integrand = funm.integrand_funm_sym_logdet(order)
                sample_fun = stochtrace.sampler_normal(y_flatten, num=num)
                estimator = stochtrace.estimator(integrand, sampler=sample_fun)
                key = jax.random.PRNGKey(333)
                logdet = estimator(matvec, key)
                #print(f"LOGDET: {logdet.item()} with order {order} and num {num}")

                integrand = funm.integrand_funm_sym(lambda x: 1./x, inv_iterations)
                residual_NTKinv_residual = integrand(matvec, residual)
                #print(f"RECONST: {residual_NTKinv_residual.item()} with {inv_iterations} iterations")

                marginal_log_lik = - 0.5 * logdet - 0.5 * residual_NTKinv_residual
                #marginal_log_lik = - 0.5 * logdet
                #marginal_log_lik = - 0.5 * residual_NTKinv_residual
            else:
                vector_params = flatten_util.ravel_pytree(params)[0]
                marginal_log_lik = -jnp.sum( vector_params**2 )

            return -marginal_log_lik, (mse,)
                                                                                
        loss_function_train = loss_function_test
        return loss_function_train, loss_function_test
    
    loss_function_train, loss_function_test = get_marginal_loss(model)

    
    preds_init = model.apply_test(params_dict['params'], x_init)
    preds_init = jax.lax.stop_gradient(preds_init)
                                                                
    ########################
    # define training step #
    if not model.has_batch_stats:
        #@jax.jit
        def train_step(opt_state, params_dict, x, y):

            projection_on_kernel_manifold = get_projection_on_kernel_manifold(x, preds_init)
            #print(f" -Distance 1 from kernel: {distance_from_kernel_manifold(params_dict['params'], x, preds_init):.7f} ")


            #projection_vector_product = get_projection_vector_product(
            #    params_dict,
            #    model,
            #    get_subset_loader(
            #        train_loader,
            #        len(train_loader.dataset),
            #        batch_size = 1
            #    ),
            #    likelihood_type = "regression"
            #)
            lanczos_iter = 30 #goood
            lanczos_iter = 6
            ggn_vector_product = get_ggn_vector_product(params_dict, model, x, likelihood_type="regression")
            eigenvec, eigenval = high_memory_lanczos(jax.random.PRNGKey(1), ggn_vector_product, P, lanczos_iter)
            i = int(0.9 * lanczos_iter)
            eigenvec = eigenvec[:,:i]
            eigenval = eigenval[:i]
            print(f"   last GGN eigenvals: {eigenval[-3:]}")
            projection_vector_product = lambda v: v  - eigenvec @ eigenvec.T @ v

            loss_fn = lambda p: loss_function_train(p, x, y)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])


            params_vector, unflatten_fun = flatten_util.ravel_pytree(params_dict['params'])
            grads_vector = flatten_util.ravel_pytree(grads)[0]
            norm_before = jnp.linalg.norm(grads_vector)
            grads_vector = projection_vector_product(grads_vector)
            norm_after = jnp.linalg.norm(grads_vector)
            print(f"    gradient norm: {norm_before.item():.3f} -> {norm_after.item():.3f}")

            params_vector = params_vector - args_dict["learning_rate"] * grads_vector
            params_dict['params'] = unflatten_fun(params_vector)


            params_dict_projected = projection_on_kernel_manifold(params_dict)
            params_projected_vector = flatten_util.ravel_pytree(params_dict_projected['params'])[0]

            print(f"   gradient step: {jnp.linalg.norm(args_dict['learning_rate']*grads_vector):.4f} - projection step {jnp.linalg.norm(params_projected_vector-params_vector):.4f}")

            print(f" -Distance 2 from kernel: {distance_from_kernel_manifold(params_dict['params'], x, preds_init):.7f} ")
            print(f" -Distance 3 from kernel: {distance_from_kernel_manifold(params_dict_projected['params'], x, preds_init):.7f} ")
            #print()
            #print()
            #print(jax.devices())
            #print(jax.devices()[0].memory_stats())
            #print()
            #print()
            loss, (acc_or_sse, ) = ret
            params_dict = params_dict_projected
            return opt_state, params_dict, loss, acc_or_sse
    elif model.has_batch_stats and not model.has_dropout:
        @jax.jit
        def train_step(opt_state, params_dict, x, y):
            loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y)
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
    elif model.has_batch_stats and model.has_dropout:
        @jax.jit
        def train_step(opt_state, params_dict, x, y, key_dropout):
            loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y, key_dropout)
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

            if model.has_dropout:
                k, key_dropout = jax.random.split(key_dropout, 2)
                opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y, k)
            else:
                opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y)

            loss += batch_loss.item()
            batch_acc_or_sse /= X.shape[0]
            acc_or_sse += batch_acc_or_sse

            if args_dict["verbose"]:
                formatted_batch_acc = f"{batch_acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{jnp.mean(batch_acc_or_sse):.3f}"]+[f"{a:.2f}" for a in batch_acc_or_sse[:10]]
                train_loader_bar.set_description(f"Epoch {epoch}/{args_dict['n_epochs']}, batch loss = {batch_loss.item():.3f}, {acc_label} = {formatted_batch_acc}")
    
        acc_or_sse /= len(train_loader)
        loss /= len(train_loader)
        params_norm = compute_norm_params(params_dict['params'])
        batch_sats_norm = compute_norm_params(params_dict['batch_stats'])
        acc_or_sse_formatted = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{jnp.mean(acc_or_sse):.3f}"]+[f"{a:.2f}" for a in acc_or_sse]
        print(f"epoch={epoch} averages - loss={loss:.3f}, params norm={params_norm:.2f}, batch_stats norm={batch_sats_norm:.2f}, {acc_label}={acc_or_sse_formatted}, time={time.time() - start_time:.3f}s")
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
                if model.has_batch_stats:
                    batch_loss, (batch_acc_or_sse, _) = loss_function_test(
                        #model, 
                        params_dict['params'], 
                        params_dict['batch_stats'], 
                        X, 
                        Y
                    )
                else:
                    batch_loss, (batch_acc_or_sse, ) = loss_function_test(
                        #model, 
                        params_dict['params'], 
                        X, 
                        Y
                    )
                loss += batch_loss.item()
                acc_or_sse += batch_acc_or_sse/X.shape[0]
            acc_or_sse = acc_or_sse/len(loader) if len(loader)>0 else 0
            loss = loss/len(loader) if len(loader)>0 else 0
            return loss, acc_or_sse, time.time() - start_time

        loss, acc_or_sse, duration = get_precise_stats(train_loader)
        acc_or_sse_formatted = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{jnp.mean(acc_or_sse):.3f}"]+[f"{a:.2f}" for a in acc_or_sse]
        print(f"Train stats\t - loss={loss:.3f}, {acc_label}={acc_or_sse_formatted}, time={duration:.3f}s")
        train_stats_dict["loss"].append(loss)
        train_stats_dict["acc_or_mse"].append(acc_or_sse)

        loss, acc_or_sse, duration = get_precise_stats(valid_loader)
        acc_or_sse_formatted = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else [f"{jnp.mean(acc_or_sse):.3f}"]+[f"{a:.2f}" for a in acc_or_sse]
        print(f"Validation stats - loss={loss:.3f}, {acc_label}={acc_or_sse_formatted} time={duration:.3f}s")
        valid_stats_dict["loss"].append(loss)
        valid_stats_dict["acc_or_mse"].append(acc_or_sse)

    epoch_stats_dict = {'epoch_'+k : v for k,v in epoch_stats_dict.items()}    
    train_stats_dict = {'train_'+k : v for k,v in train_stats_dict.items()}    
    valid_stats_dict = {'valid_'+k : v for k,v in valid_stats_dict.items()}    
    stats_dict = {**train_stats_dict, **valid_stats_dict, **epoch_stats_dict}




    ntk_matvec = get_ntk_vector_product(params_dict, model, x_init, likelihood_type="regression")
    matvec = lambda v: 1. * v + 1. * ntk_matvec(v)
    identity = jnp.eye(100)
    ntk = jax.vmap(matvec)(identity)
    resss = model.apply_test(params_dict['params'],x_init).reshape(-1) - y_init.reshape(-1)
    log_det = jax.numpy.linalg.slogdet(ntk)[1]
    ntk_inv_resss = jax.scipy.linalg.solve(ntk, resss)
    true_marginal_log_lik = - 0.5 * log_det - 0.5 * resss @ ntk_inv_resss
    print(f"log det: {log_det} - residual: {resss @ ntk_inv_resss}")
    print(f"THE TRUE ONE: {true_marginal_log_lik}")






    return params_dict, stats_dict