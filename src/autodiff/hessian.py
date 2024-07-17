import jax
import jax.numpy as jnp
import flax
import functools
from typing import Tuple
from jax import flatten_util

from src.training.loss import log_gaussian_log_loss, cross_entropy_loss, multiclass_binary_cross_entropy_loss

import time

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
    elif likelihood_type == "binary_multiclassification":
        negative_log_likelihood = multiclass_binary_cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
    
    params = params_dict['params']
    if not model.has_batch_stats:
        loss_on_data = lambda p: negative_log_likelihood(model.apply_test(p, X), Y)
    else:
        batch_stats = params_dict['batch_stats']
        loss_on_data = lambda p: negative_log_likelihood(
            model.apply_test(p, batch_stats, X), 
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


def get_hessian_vector_product_dataloader(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression"
    ):

    if likelihood_type == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood_type == "classification":
        negative_log_likelihood = cross_entropy_loss
    elif likelihood_type == "binary_multiclassification":
        negative_log_likelihood = functools.partial(multiclass_binary_cross_entropy_loss, class_frequencies=dataloader.dataset.dataset.dataset.class_frequencies)
    else:
        raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
    
    params = params_dict['params']
    if not model.has_batch_stats:
        loss_apply = lambda x, y, p: negative_log_likelihood(model.apply_test(p, x), y)
    else:
        batch_stats = params_dict['batch_stats']
        loss_apply = lambda x, y, p: negative_log_likelihood(model.apply_test(p, batch_stats, x), y)
    devectorize_fun = flatten_util.ravel_pytree(params)[1]
    flatten_param = jnp.array(flatten_util.ravel_pytree(params)[0])

    @jax.jit
    def hessian_tree_product_batch(X, Y, tree):
        loss_on_data = functools.partial(loss_apply, X, Y)
        return jax.jvp(jax.jacrev(loss_on_data), (params,), (tree,))[1]
    
    @jax.jit
    def hessian_vector_product_batch(X, Y, v):
        tree = devectorize_fun(v)
        ggn_tree = hessian_tree_product_batch(X, Y, tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.asarray(ggn_v)

    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())
    y_init = jnp.asarray(batch[1].numpy())

    start = time.time()
    hessian_vector_product_batch(x_init, y_init, flatten_param)
    print(f"One BATCH HESSIAN vp took {time.time()-start} seconds")
    start = time.time()
    hessian_vector_product_batch(x_init, y_init, jnp.ones_like(flatten_param))
    print(f"Again...... it took {time.time()-start} seconds")
    start = time.time()
    hessian_vector_product_batch(x_init, y_init, 2*jnp.ones_like(flatten_param))
    print(f"Aaand again...... it took {time.time()-start} seconds")

    
    def hessian_vector_product_dataloader(v):
        """
        The loop in this function will never be jitted,
        even inside a jit(), so we should be fine with
        file system access etc.
        """
        result = jnp.zeros_like(v)
        for batch in dataloader:
            X = jnp.asarray(batch[0].numpy())
            Y = jnp.asarray(batch[1].numpy())
            result_batch = hessian_vector_product_batch(X, Y, v)
            result += result_batch
        return result
    
    result_shape = jax.ShapeDtypeStruct(flatten_param.shape, flatten_param.dtype)
    def hessian_vector_product(v):
        return jax.pure_callback(hessian_vector_product_dataloader, result_shape, v)
    
    return jax.jit(hessian_vector_product)


#################################################
# Instatiate hessian of loss wrt network output #

def get_sqrt_hessian_loss_explicit(params_dict, model, likelihood_type = "regression", output_dim=None):

    params = params_dict['params']
    if not model.has_batch_stats:
        model_on_params = lambda data: model.apply_test(params, data)
    else:
        batch_stats = params_dict['batch_stats']
        model_on_params = lambda data: model.apply_test(params, batch_stats, data)

    if likelihood_type == "regression" or likelihood_type == "binary_multiclassification":
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