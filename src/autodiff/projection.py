import jax
import jax.numpy as jnp
import flax
import functools
from jax import flatten_util

from src.training.losses import get_loss_function

import time


#######################################
# Generalize Gauss Newtown projection #

def get_projection_vector_product(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression"
    ):

    _, loss_function_test = get_loss_function(
        model,
        likelihood = likelihood_type,
        class_frequencies = dataloader.dataset.dataset.class_frequencies if likelihood_type=="binary_multiclassification" else None
    )

    params = params_dict['params']
    if not model.has_batch_stats:
        loss_apply = lambda x, y, p: loss_function_test(p, x, y)
    else:
        batch_stats = params_dict['batch_stats']
        loss_apply = lambda x, y, p: loss_function_test(p, batch_stats, x, y)
    devectorize_fun = flatten_util.ravel_pytree(params)[1]
    flatten_param = jnp.asarray(flatten_util.ravel_pytree(params)[0])
    
    @jax.jit
    def single_datapoint_projection_vector_product(X, Y, v):
        loss_on_data = functools.partial(loss_apply, X, Y)
        ret, grad = jax.value_and_grad(loss_on_data, has_aux=True)(params_dict['params'])
        loss, _ = ret
        grad = jnp.asarray(flatten_util.ravel_pytree(grad)[0])

        projection = (jnp.dot(v,grad) / jnp.dot(grad,grad)) * grad
        #print(jnp.dot(v,v), "\t", jnp.dot(projection,projection), "\t", jnp.dot(grad,grad))
        #print("\t",jnp.dot(v - projection, grad))
        return v - projection

    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())
    y_init = jnp.asarray(batch[1].numpy())

    start = time.time()
    single_datapoint_projection_vector_product(x_init, y_init, flatten_param)
    print(f"One BATCH PROJECTION vp took {time.time()-start} seconds")
    start = time.time()
    single_datapoint_projection_vector_product(x_init, y_init, jnp.ones_like(flatten_param))
    print(f"Again...... it took {time.time()-start} seconds")
    start = time.time()
    single_datapoint_projection_vector_product(x_init, y_init, 2*jnp.ones_like(flatten_param))
    print(f"Aaand again...... it took {time.time()-start} seconds")

    
    def projection_vector_product(v):
        """
        The loop in this function will never be jitted,
        even inside a jit(), so we should be fine with
        file system access etc.
        """
        i = 0
        norms = []
        for batch in dataloader:
            X = jnp.asarray(batch[0].numpy())
            Y = jnp.asarray(batch[1].numpy())
            v = single_datapoint_projection_vector_product(X, Y, v)
            i+=1
            if not i%1000:
                norms.append(jnp.dot(v,v))
                #print(jnp.dot(v,v))
        return v, norms
    
    result_shape = jax.ShapeDtypeStruct(flatten_param.shape, flatten_param.dtype)
    #def pure_callback_projection_vector_product(v):
    #    return jax.pure_callback(projection_vector_product, result_shape, v)
    
    #return jax.jit(pure_callback_projection_vector_product)
    #return pure_callback_projection_vector_product
    return projection_vector_product