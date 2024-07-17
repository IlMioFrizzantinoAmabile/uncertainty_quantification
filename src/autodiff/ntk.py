import jax
import jax.numpy as jnp
import flax


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
    if not model.has_batch_stats:
        model_on_data = lambda p: model.apply_test(p, data_array)
    else:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p, data: model.apply_test(p, batch_stats, data_array)

    @jax.jit
    def ntk_vector_product(v):
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
        
        return sqrtH_JJt_sqrtH_v.reshape(-1)
    
    return ntk_vector_product