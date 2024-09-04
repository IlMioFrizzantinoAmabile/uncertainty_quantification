import jax
import jax.numpy as jnp
from functools import partial
from typing import Literal

from src.models.wrapper import Model

def mse_loss(preds, y):
    residual = preds.reshape(-1) - y.reshape(-1)
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
def multiclass_binary_cross_entropy_loss(preds, y, class_frequencies, rho=1.0):
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
    #bce = jnp.einsum("bo, o -> bo", y * log_p, 1./class_frequencies) + (1. - y) * log_not_p
    bce = -(y * log_p + (1. - y) * log_not_p)
    return jnp.mean(bce)


def get_loss_function(
        model: Model,
        likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification",
        class_frequencies = None
    ):

    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
        extra_stats_function = lambda preds, y : jnp.sum((preds-y)**2)                                  # sum of squared error
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
        extra_stats_function = lambda preds, y : jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))    # accuracy
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = lambda preds, y: multiclass_binary_cross_entropy_loss(preds, y, class_frequencies)
        extra_stats_function = lambda preds, y : jnp.sum((preds>0.) == (y==1), axis=0)                  # multiclass accuracy
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")

    if not model.has_batch_stats and not model.has_dropout:
        @jax.jit
        def loss_function_train(params, x, y):
            preds = model.apply_train(params, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, )
        @jax.jit
        def loss_function_test(params, x, y):
            preds = model.apply_test(params, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, )
        
    elif not model.has_batch_stats and model.has_dropout:
        @jax.jit
        def loss_function_train(params, x, y, key_dropout):
            preds = model.apply_train(params, x, key_dropout)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, )
        @jax.jit
        def loss_function_test(params, x, y):
            preds = model.apply_test(params, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, )
        
    elif model.has_batch_stats and not model.has_dropout:
        @jax.jit
        def loss_function_train(params, batch_stats, x, y):
            preds, new_model_state = model.apply_train(params, batch_stats, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, new_model_state)
        @jax.jit
        def loss_function_test(params, batch_stats, x, y):
            preds = model.apply_test(params, batch_stats, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, None)
        
    elif model.has_batch_stats and model.has_dropout:
        @jax.jit
        def loss_function_train(params, batch_stats, x, y, key_dropout):
            preds, new_model_state = model.apply_train(params, batch_stats, x, key_dropout)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, new_model_state)
        @jax.jit
        def loss_function_test(params, batch_stats, x, y):
            preds = model.apply_test(params, batch_stats, x)
            loss = negative_log_likelihood(preds, y)
            acc_or_sse = extra_stats_function(preds, y)
            return loss, (acc_or_sse, None)

            
    return loss_function_train, loss_function_test


def get_likelihood(
        likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification",
        class_frequencies = None
    ):

    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
        extra_stats_function = lambda preds, y : jnp.sum((preds-y)**2)                                  # sum of squared error
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
        extra_stats_function = lambda preds, y : jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))    # accuracy
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = lambda preds, y: multiclass_binary_cross_entropy_loss(preds, y, class_frequencies)
        extra_stats_function = lambda preds, y : jnp.sum((preds>0.) == (y==1), axis=0)                  # multiclass accuracy
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
    
    return negative_log_likelihood, extra_stats_function