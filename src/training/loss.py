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


@partial(jax.jit, static_argnames=['rho'])
def multiclass_binary_cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, n_classes) (logits) Each element is the unnormalized log probability of a binary
      prediction.
    y: (batch_size, n_classes) Binary labels whose values are {0,1} or multi-class target
      probabilities. See note about compatibility with `logits` above.
    """
    preds = preds * rho
    y = y.astype(preds.dtype)
    log_p = jax.nn.log_sigmoid(preds)
    log_not_p = jax.nn.log_sigmoid(-preds)
    return jnp.mean(y * log_p + (1. - y) * log_not_p)

#@partial(jax.jit, static_argnames=['model', 'train', 'likelihood'])
def calculate_loss_with_batchstats(
    model, 
    params, 
    batch_stats, 
    x, 
    y, 
    train: bool = False,
    likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification"
):
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = multiclass_binary_cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
    outs = model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                train=train,
                mutable=['batch_stats'] if train else False)
    preds, new_model_state = outs if train else (outs, None)
    loss = negative_log_likelihood(preds, y)
    if likelihood == "regression":
        sse = jnp.sum((preds-y)**2)
        return loss, (sse, new_model_state)
    elif likelihood == "classification":
        acc = jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))
        return loss, (acc, new_model_state)
    elif likelihood == "binary_multiclassification":
        num_classes = preds.shape[1]
        acc = jnp.sum((preds>0.) == (y==1)) / num_classes
        return loss, (acc, new_model_state)


@partial(jax.jit, static_argnames=['model', 'likelihood'])
def calculate_loss_without_batchstats(
    model, 
    params, 
    x, 
    y,
    likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification",
):
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = multiclass_binary_cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
    preds = model.apply(params, x)
    loss = negative_log_likelihood(preds, y)

    if likelihood == "regression":
        sse = jnp.sum((preds-y)**2)
        return loss, (sse, )
    elif likelihood == "classification":
        acc = jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))
        return loss, (acc, )
    elif likelihood == "binary_multiclassification":
        num_classes = preds.shape[1]
        acc = jnp.sum((preds>0.) == (y==1)) / num_classes
        return loss, (acc, )