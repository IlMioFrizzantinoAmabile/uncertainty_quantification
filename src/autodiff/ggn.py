import jax
import jax.numpy as jnp
import flax
from typing import Tuple
from jax import flatten_util

from src.models.utils import has_batchstats


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
    if not has_batchstats(model):
        model_on_data = lambda p: model.apply(p, data_array)
        devectorize_fun = flatten_util.ravel_pytree(params)[1]
    else:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p: model.apply(
                    {'params': p, 'batch_stats': batch_stats}, 
                    data_array,
                    train=False,
                    mutable=False
                )
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