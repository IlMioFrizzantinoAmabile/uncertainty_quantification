import jax
import jax.numpy as jnp
from functools import partial
from typing import Literal

def mse_loss(preds, y):
    residual = preds - y
    return jnp.mean(residual**2)

@partial(jax.jit, static_argnames=['rho'])
def log_gaussian_log_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, output_dim) (predictions)
    y: (batch_size, output_dim) (targets)
    """
    O = y.shape[-1]
    return 0.5 * O * jnp.log(2 * jnp.pi) - 0.5 * O * jnp.log(rho) + 0.5 * rho * mse_loss(preds, y)

@partial(jax.jit, static_argnames=['rho'])
def cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, n_classes) (logits)
    y: (batch_size, n_classes) (one-hot labels)
    """
    preds = preds * rho
    preds = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.mean(jnp.sum(preds * y, axis=-1))

@partial(jax.jit, static_argnames=['model', 'train', 'likelihood'])
def calculate_loss_with_batchstats(
    model, 
    params, 
    batch_stats, 
    x, 
    y, 
    train: bool = False,
    likelihood: Literal["classification", "regression"] = "classification"
):
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression' or 'classification'.")
    outs = model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                train=train,
                mutable=['batch_stats'] if train else False)
    preds, new_model_state = outs if train else (outs, None)
    loss = negative_log_likelihood(preds, y)
    if likelihood == "classification":
        acc = jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))
        return loss, (acc, new_model_state)
    elif likelihood == "regression":
        sse = jnp.sum((preds-y)**2)
        return loss, (sse, new_model_state)


@partial(jax.jit, static_argnames=['model', 'likelihood'])
def calculate_loss_without_batchstats(
    model, 
    params, 
    x, 
    y,
    likelihood: Literal["classification", "regression"] = "classification"
):
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression' or 'classification'.")
    preds = model.apply(params, x)
    loss = negative_log_likelihood(preds, y)

    if likelihood == "classification":
        acc = jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))
        return loss, (acc, )
    elif likelihood == "regression":
        sse = jnp.sum((preds-y)**2)
        return loss, (sse, )


def compute_num_params(params):
    vector_params = jax.flatten_util.ravel_pytree(params)[0]
    return vector_params.shape[0]

def compute_norm_params(params):
    vector_params = jax.flatten_util.ravel_pytree(params)[0]
    return jnp.linalg.norm(vector_params).item()