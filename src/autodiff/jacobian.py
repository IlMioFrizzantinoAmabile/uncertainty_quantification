import jax
import jax.numpy as jnp
from jax import flatten_util
import flax

from src.models.utils import has_batchstats


###################################
# Jacobian products without trees #

def get_jacobian_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        single_datapoint = False,
    ):
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)
    
    params = params_dict['params']
    if not has_batchstats(model):
        model_on_data = lambda p: model.apply(p, data_array)
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
    def jacobian_vector_product(vector):
        # parameter space -> data times output space
        tree = devectorize_fun(vector)
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        return J_tree.reshape(-1)
    
    return jacobian_vector_product
    
def get_jacobianT_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        single_datapoint = False,
    ):
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)
    B = data_array.shape[0]
    
    params = params_dict['params']
    if not has_batchstats(model):
        model_on_data = lambda p: model.apply(p, data_array)
    else:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p: model.apply(
            {'params': p, 'batch_stats': batch_stats}, 
            data_array,
            train=False,
            mutable=False
        )
    _, model_on_data_vjp = jax.vjp(model_on_data, params)
    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]

    @jax.jit
    def jacobianT_vector_product(vector):
        # data times output space -> parameter space 
        vector = vector.reshape((B, -1))
        Jt_vector = model_on_data_vjp(vector)[0]
        return vectorize_fun(Jt_vector)
    
    return jacobianT_vector_product



#######################################
# Instatiate full jacobian explicitly #

def get_jacobian_explicit(params_dict, model):
    vectorize_fun = lambda x : flatten_util.ravel_pytree(x)[0]

    params = params_dict['params']
    if not has_batchstats(model):
        model_fun = lambda p, data: model.apply(p, data)
    else:
        batch_stats = params_dict['batch_stats']
        model_fun = lambda p, data: model.apply(
            {'params': p, 'batch_stats': batch_stats}, 
            data,
            train=False,
            mutable=False
        )

    @jax.jit
    def jacobian(query_data):
        query_data = jnp.expand_dims(query_data, 0)
        fun = lambda p : model_fun(p, query_data)
        #pytree_jacob = jax.jacfwd(fun)(params)
        pytree_jacob = jax.jacrev(fun)(params)
        # we return the jacobian as a model.output_dim x p matrix
        # where p is the number of params
        jacob_array = jnp.asarray([vectorize_fun(jax.tree_map(
            lambda x: x[:, i,:], pytree_jacob)) for i in range(model.output_dim)]) 
        return jacob_array
    return jacobian