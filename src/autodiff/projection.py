import jax
import jax.numpy as jnp
import flax
import functools
from jax import flatten_util

from src.training.losses import get_likelihood

import time


#######################################
# Generalize Gauss Newtown projection #

def get_loss_projection_vector_product(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression",
        num_epochs: int = 1,
    ):

    negative_log_likelihood, _ = get_likelihood(
        likelihood = likelihood_type,
        class_frequencies = dataloader.dataset.dataset.class_frequencies if likelihood_type=="binary_multiclassification" else None
    )

    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_apply = lambda data, p: model.apply_test(p, attention_mask, relative_position_index, data)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_apply = lambda data, p: model.apply_test(p, batch_stats, data)
    else:
        model_apply = lambda data, p: model.apply_test(p, data)
    loss_apply = lambda x, y, p: negative_log_likelihood(model_apply(x, p), y)
    params = params_dict['params']
    devectorize_fun = flatten_util.ravel_pytree(params)[1]
    flatten_param = jnp.asarray(flatten_util.ravel_pytree(params)[0])
    
    @jax.jit
    def single_datapoint_projection_vector_product(X, Y, v):
        loss_on_data = functools.partial(loss_apply, X, Y)
        #ret, grad = jax.value_and_grad(loss_on_data, has_aux=True)(params_dict['params'])
        #loss, _ = ret
        loss, grad = jax.value_and_grad(loss_on_data)(params_dict['params'])
        grad = jnp.asarray(flatten_util.ravel_pytree(grad)[0])

        projection = (jnp.dot(v,grad) / jnp.dot(grad,grad)) * grad
        #print(jnp.dot(v,v), "\t", jnp.dot(projection,projection), "\t", jnp.dot(grad,grad))
        #print("\t",jnp.dot(v - projection, grad))
        return v - projection

    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())
    y_init = jnp.asarray(batch[1].numpy())

    #start = time.time()
    single_datapoint_projection_vector_product(x_init, y_init, flatten_param)
    #print(f"One BATCH PROJECTION vp took {time.time()-start} seconds")
    #start = time.time()
    #single_datapoint_projection_vector_product(x_init, y_init, jnp.ones_like(flatten_param))
    #print(f"Again...... it took {time.time()-start} seconds")
    #start = time.time()
    #single_datapoint_projection_vector_product(x_init, y_init, 2*jnp.ones_like(flatten_param))
    #print(f"Aaand again...... it took {time.time()-start} seconds")

    
    def projection_vector_product(v):
        """
        The loop in this function will never be jitted,
        even inside a jit(), so we should be fine with
        file system access etc.
        """
        #i = 0
        #norms = []
        for e in range(num_epochs):
            for batch in dataloader:
                X = jnp.asarray(batch[0].numpy())
                Y = jnp.asarray(batch[1].numpy())
                v = single_datapoint_projection_vector_product(X, Y, v)
                #i+=1
                #if not i%1000:
                #    norms.append(jnp.dot(v,v))
                    #print(jnp.dot(v,v))
        print("projection done")
        return v#, norms
    
    result_shape = jax.ShapeDtypeStruct(flatten_param.shape, flatten_param.dtype)
    def pure_callback_projection_vector_product(v):
        return jax.pure_callback(projection_vector_product, result_shape, v)
    
    return jax.jit(pure_callback_projection_vector_product)
    #return pure_callback_projection_vector_product
    #return projection_vector_product



def get_projection_vector_product(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression",
        num_epochs: int = 1,
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


    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())
    y_init = jnp.asarray(batch[1].numpy())
    B, O = y_init.shape

    identity = jnp.eye(B * O)
    
    @jax.jit
    def ntk_vector_product(X, v):
        v = v.reshape((B, O))

        model_on_data = functools.partial(model_apply, X)
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
        
        sqrtH_JJt_sqrtH_v = sqrtH_JJt_sqrtH_v.reshape(B*O)
        return sqrtH_JJt_sqrtH_v
    

    @jax.jit
    def single_batch_projection_vector_product(X, v):

        # instatiate the ntk
        ntk_on_data_vector_product = functools.partial(ntk_vector_product, X)
        ntk = jax.vmap(ntk_on_data_vector_product)(identity)

        # compute pseudo inverse
        eigvals, eigvecs = jnp.linalg.eigh(ntk)
        unstable_idx = eigvals < 1e-5
        inv_eigvals = jnp.where(unstable_idx, 1., eigvals)      # set the unstable index to 1
        inv_eigvals = 1./inv_eigvals                            # invert the eigenvalues
        inv_eigvals = jnp.where(unstable_idx, 0., inv_eigvals)  # set the unstable index to 0
        inv_ntk = (eigvecs * inv_eigvals) @ eigvecs.T
         

        model_on_data = functools.partial(model_apply, X)
        tree = devectorize_fun(v)
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))

        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        if likelihood_type == "regression":
            # sqrtH = identity
            sqrtHJ_tree = J_tree
            ntk_sqrtHJ_tree = (inv_ntk @ sqrtHJ_tree.reshape(B*O)).reshape((B,O)) 
            sqrtH_ntk_sqrtHJ_tree = ntk_sqrtHJ_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            pred = jnp.sqrt(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            sqrtH = D - H

            sqrtHJ_tree = jnp.einsum("boi, bi->bo", sqrtH, J_tree)
            ntk_sqrtHJ_tree = (inv_ntk @ sqrtHJ_tree.reshape(-1)).reshape((B,O)) 
            sqrtH_ntk_sqrtHJ_tree = jnp.einsum("boi, bi->bo", sqrtH, ntk_sqrtHJ_tree)
        elif likelihood_type == "binary_multiclassification":
            pred = jax.nn.sigmoid(pred)
            pred = jax.lax.stop_gradient(pred)
            pred = jnp.sqrt(pred)
            sqrtH = pred - pred**2

            sqrtHJ_tree = jnp.einsum("bo, bo->bo", pred - pred**2, J_tree)
            ntk_sqrtHJ_tree = (inv_ntk @ sqrtHJ_tree.reshape(-1)).reshape((B,O)) 
            sqrtH_ntk_sqrtHJ_tree = jnp.einsum("bo, bo->bo", pred - pred**2, ntk_sqrtHJ_tree)
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression', 'classification' or 'binary_multiclassification.")
        
        JtsqrtH_ntk_sqrtHJ_tree = model_on_data_vjp(sqrtH_ntk_sqrtHJ_tree)[0]
        JtsqrtH_ntk_sqrtHJ_v = flatten_util.ravel_pytree(JtsqrtH_ntk_sqrtHJ_tree)[0]

        #print(jnp.sum(v ** 2), jnp.sum(JtsqrtH_ntk_sqrtHJ_v ** 2))
        return v - JtsqrtH_ntk_sqrtHJ_v
        #return jnp.asarray(v - JtsqrtH_ntk_sqrtHJ_v)


    start = time.time()
    single_batch_projection_vector_product(jnp.ones_like(x_init), jnp.ones_like(flatten_param))
    print(f"One BATCH PROJECTION vp took {time.time()-start} seconds")
    start = time.time()
    single_batch_projection_vector_product(x_init, flatten_param)
    print(f"Again...... it took {time.time()-start} seconds")

    
    def projection_vector_product(v):
        """
        The loop in this function will never be jitted,
        even inside a jit(), so we should be fine with
        file system access etc.
        """
        for e in range(num_epochs):
            for batch in dataloader:
                X = jnp.asarray(batch[0].numpy())
                start = time.time()
                v = single_batch_projection_vector_product(X, v)
                #print(f".... inside the loop it took {time.time()-start} seconds")
                #print("AAAA", jnp.sum(v**2))
        print("projection done")
        return v
    
    result_shape = jax.ShapeDtypeStruct(flatten_param.shape, flatten_param.dtype)
    def pure_callback_projection_vector_product(v):
        return jax.pure_callback(projection_vector_product, result_shape, v)
    
    #return projection_vector_product
    return jax.jit(pure_callback_projection_vector_product)