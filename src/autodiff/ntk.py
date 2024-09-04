import jax
import jax.numpy as jnp
import flax

from matfree.backend import tree_util


#####################################
# Neural Tangent Kernel products #

def get_ntk_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array,
        single_datapoint = False,
        likelihood_type: str = "regression"
    ):
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)
    B = data_array.shape[0]
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

    @jax.jit
    def ntk_vector_product(v):
        #print("EI! I'm Marco's matvec, I'm receiving a vector that looks like", v.shape)
        #_, unflatten = tree_util.ravel_pytree(v)
        #v = unflatten(v)
        #print("a", v.shape)
        #v, _ = tree_util.ravel_pytree(v)
        #print("b", v.shape)

        v = v.reshape((B, -1))
                        
        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        if likelihood_type == "regression":
            sqrtH_v = v
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            pred = jnp.sqrt(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            sqrtH = D - H
            sqrtH_v = jnp.einsum("boi, bi->bo", sqrtH, v)
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
        
        Jt_sqrtH_v = model_on_data_vjp(sqrtH_v)[0]
        _, JJt_sqrtH_v = jax.jvp(model_on_data, (params,), (Jt_sqrtH_v,))

        if likelihood_type == "regression":
            sqrtH_JJt_sqrtH_v = JJt_sqrtH_v
        elif likelihood_type == "classification":
            sqrtH_JJt_sqrtH_v = jnp.einsum("boi, bi->bo", sqrtH, JJt_sqrtH_v)
        
        sqrtH_JJt_sqrtH_v = sqrtH_JJt_sqrtH_v.reshape(-1)
        #print("EI! I'm Marco's matvec, I'm returning a vector that looks like", sqrtH_JJt_sqrtH_v.shape)
        return sqrtH_JJt_sqrtH_v
    
    return ntk_vector_product