import jax
import jax.numpy as jnp
import flax
from typing import Tuple
from jax import flatten_util

from src.models.utils import has_batchstats

from src.training.loss import log_gaussian_log_loss, cross_entropy_loss


###################
# Hessian product #

def get_hessian_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: Tuple[jax.Array],
        single_datapoint = False,
        likelihood_type: str = "regression"
    ):
    X, Y = data_array
    if single_datapoint:
        X = jnp.expand_dims(X, 0)
        Y = jnp.expand_dims(Y, 0)
    if likelihood_type == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood_type == "classification":
        negative_log_likelihood = cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
    
    params = params_dict['params']
    if not has_batchstats(model):
        loss_on_data = lambda p: negative_log_likelihood(model.apply(p, X), Y)
        devectorize_fun = flatten_util.ravel_pytree(params)[1]
    else:
        batch_stats = params_dict['batch_stats']
        loss_on_data = lambda p: negative_log_likelihood(
            model.apply(
                    {'params': p, 'batch_stats': batch_stats}, 
                    X,
                    train=False,
                    mutable=False
                ), 
            Y)
        devectorize_fun = flatten_util.ravel_pytree(params)[1]
    @jax.jit
    def hessian_tree_product(tree):
        return jax.jvp(jax.jacrev(loss_on_data), (params,), (tree,))[1]
    @jax.jit
    def hessian_vector_product(v):
        tree = devectorize_fun(v)
        hessian_tree = hessian_tree_product(tree)
        hessian_v = jax.flatten_util.ravel_pytree(hessian_tree)[0]
        return jnp.array(hessian_v)
    return hessian_vector_product


#################################################
# Instatiate hessian of loss wrt network output #

def get_sqrt_hessian_loss_explicit(params_dict, model, likelihood_type = "regression", output_dim=None):

    params = params_dict['params']
    if not has_batchstats(model):
        model_on_params = lambda data: model.apply(params, data)
    else:
        batch_stats = params_dict['batch_stats']
        model_on_params = lambda data: model.apply(
            {'params': params, 'batch_stats': batch_stats}, 
            data,
            train=False,
            mutable=False
        )

    if likelihood_type == "regression":
        if output_dim is None:
            @jax.jit
            def sqrt_hessian_loss(query_data):
                query_data = jnp.expand_dims(query_data, 0)
                pred = model_on_params(query_data)
                return jnp.eye(pred.shape[-1])
        else:
            @jax.jit
            def sqrt_hessian_loss(query_data):
                return jnp.eye(output_dim)
    elif likelihood_type == "classification":
        @jax.jit
        def sqrt_hessian_loss(query_data):
            query_data = jnp.expand_dims(query_data, 0)
            pred = model_on_params(query_data)
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            return H[0]
        
    return sqrt_hessian_loss