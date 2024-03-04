import jax
import jax.numpy as jnp

def get_diagonal(
    matrix_vector_product,
    dim,
    sequential = True,
    n_samples: int = None,
    key: jax.random.PRNGKey = None,
    estimator = "Rademacher",
):
    if n_samples is not None and n_samples >= dim:
        print(f"You called stochastich Hutchinson on dim {dim} with {n_samples} samples, you dumb. I'm switching to deterministic, so it's faster and more accurate")
        n_samples = None

    if n_samples is None:
        # deterministic
        n_samples = dim
        base_vectors = jnp.concatenate((jnp.zeros((dim-1,)), jnp.ones((1,)), jnp.zeros((dim-1,))))
        get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (dim-k-1,), (dim,))
        aggr_fun = (lambda x, y : x + y) if sequential else jax.numpy.sum
    else:
        #stochastic
        if key is None:
            raise ValueError("Stochastic estimator needs a random key")
        keys = jax.random.split(key, n_samples)
        if estimator == "Rademacher":
            get_vec = lambda k: jax.random.bernoulli(keys[k], p=0.5, shape=(dim,)).astype(float) * 2 - 1
        elif estimator == "Normal":
            get_vec = lambda k: jax.random.normal(keys[k], shape=(dim,))
        aggr_fun = (lambda x, y : x + y/n_samples) if sequential else jax.numpy.mean

    def one_sample_diagonal(k):
        vec = get_vec(k)
        M_vec = matrix_vector_product(vec)
        diagonal = vec * M_vec
        return diagonal

    if sequential:
        return jax.lax.fori_loop(
            0, n_samples, 
            lambda k, temp : aggr_fun(temp, one_sample_diagonal(k)), 
            jnp.zeros((dim,))
        )
    else:
        diagonals = jax.vmap(one_sample_diagonal)(jnp.arange(n_samples)) 
        return aggr_fun(diagonals, axis=0) 
    

def get_trace(
    matrix_vector_product,
    dim,
    sequential = True,
    n_samples: int = None,
    key: jax.random.PRNGKey = None,
    estimator = "Rademacher",
):
    if n_samples is not None and n_samples >= dim:
        print(f"You called stochastich Hutchinson on dim {dim} with {n_samples} samples, you dumb. I'm switching to deterministic, so it's faster and accurate")
        n_samples = None

    if n_samples is None:
        # deterministic
        n_samples = dim
        base_vectors = jnp.concatenate((jnp.zeros((dim-1,)), jnp.ones((1,)), jnp.zeros((dim-1,))))
        get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (dim-k-1,), (dim,))
        aggr_fun = (lambda x, y : x + y) if sequential else jax.numpy.sum
    else:
        #stochastic
        if key is None:
            raise ValueError("Stochastic estimator needs a random key")
        keys = jax.random.split(key, n_samples)
        if estimator == "Rademacher":
            get_vec = lambda k: jax.random.bernoulli(keys[k], p=0.5, shape=(dim,)).astype(float) * 2 - 1
        elif estimator == "Normal":
            get_vec = lambda k: jax.random.normal(keys[k], shape=(dim,))
        aggr_fun = (lambda x, y : x + y/n_samples) if sequential else jax.numpy.mean

    @jax.jit
    def one_sample_trace(k):
        vec = get_vec(k)
        M_vec = matrix_vector_product(vec)
        trace = vec @ M_vec
        return trace

    if sequential:
        return jax.lax.fori_loop(
            0, n_samples, 
            lambda k, temp : aggr_fun(temp, one_sample_trace(k)), 
            0.
        )
    else:
        traces = jax.vmap(one_sample_trace)(jnp.arange(n_samples)) 
        return aggr_fun(traces, axis=0) 