import jax
import jax.numpy as jnp
import flax
import functools
from jax import flatten_util

import time


#####################################
# Generalize Gauss Newtown products #

def get_ggn_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array,
        single_datapoint = False,
        likelihood_type: str = "regression"
    ):
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)

    params = params_dict['params']
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_on_data = lambda p: model.apply_test(p, attention_mask, relative_position_index, data_array)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p: model.apply_test(p, batch_stats, data_array)
    else:
        model_on_data = lambda p: model.apply_test(p, data_array)
    devectorize_fun = flatten_util.ravel_pytree(params)[1]

    @jax.jit
    def ggn_tree_product(tree):
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        if likelihood_type == "regression":
            HJ_tree = J_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
        elif likelihood_type == "binary_multiclassification":
            #pred = jax.nn.sigmoid(pred)
            #HJ_tree = jnp.einsum("bo, bo->bo", pred - pred**2, J_tree)
            HJ_tree = J_tree
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree
    @jax.jit
    def ggn_vector_product(v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product(tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.array(ggn_v)
    return ggn_vector_product



def get_ggn_vector_product_dataloader(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression"
    ):
    params = params_dict['params']
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_apply = lambda data, p: model.apply_test(p, attention_mask, relative_position_index, data)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_apply = lambda data, p: model.apply_test(p, batch_stats, data)
    else:
        model_apply = lambda data, p: model.apply_test(p, data)
    devectorize_fun = flatten_util.ravel_pytree(params)[1]
    flatten_param = jnp.array(flatten_util.ravel_pytree(params)[0])

    @jax.jit
    def ggn_tree_product_batch(X, tree):
        #model_on_data = lambda p: model_apply(p, X)
        model_on_data = functools.partial(model_apply, X)
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        if likelihood_type == "regression":
            HJ_tree = J_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
        elif likelihood_type == "binary_multiclassification":
            #pred = jax.nn.sigmoid(pred)
            #HJ_tree = jnp.einsum("bo, bo->bo", pred - pred**2, J_tree)
            HJ_tree = J_tree
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree
    
    @jax.jit
    def ggn_vector_product_batch(X, v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product_batch(X, tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.asarray(ggn_v)

    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())

    start = time.time()
    ggn_vector_product_batch(jnp.ones_like(x_init), 2*jnp.ones_like(flatten_param))
    print(f"One BATCH GGN vp took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product_batch(x_init, flatten_param)
    print(f"Again...... it took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product_batch(x_init, jnp.ones_like(flatten_param))
    print(f"Aaand again...... it took {time.time()-start} seconds")

    
    def ggn_vector_product_dataloader(v):
        """
        The loop in this function will never be jitted,
        even inside a jit(), so we should be fine with
        file system access etc.
        """
        result = jnp.zeros_like(v)
        for batch in dataloader:
            #print("batch")
            X = jnp.asarray(batch[0].numpy())
            #start = time.time()
            result_batch = ggn_vector_product_batch(X, v)
            result += result_batch
            #print(f".... inside the loop it took {time.time()-start} seconds")
        return result
    
    result_shape = jax.ShapeDtypeStruct(flatten_param.shape, flatten_param.dtype)
    def ggn_vector_product(v):
        return jax.pure_callback(ggn_vector_product_dataloader, result_shape, v)

    #return ggn_vector_product_dataloader  
    return jax.jit(ggn_vector_product)  