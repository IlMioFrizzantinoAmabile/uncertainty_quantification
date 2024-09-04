import jax
import jax.numpy as jnp

def get_frobenius_norm(
    matrix_vector_product,
    dim_in,
    sequential = False,
):
    base_vectors = jnp.concatenate((jnp.zeros((dim_in-1,)), jnp.ones((1,)), jnp.zeros((dim_in-1,))))
    get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (dim_in-k-1,), (dim_in,))
    def get_norm(i):
        e_i = get_vec(i)
        col_i = matrix_vector_product(e_i)
        return jnp.sum(col_i ** 2)
    if sequential:
        return jax.lax.fori_loop(
            0, dim_in, 
            lambda i, temp : temp + get_norm(i), 
            0.
        )
    else:
        # vmap the computation wrt the input dimension
        return jnp.sum(jax.vmap(get_norm)(jnp.arange(dim_in)))
    

def get_frobenius_norm_sequential(
    matrix_vector_product,
    dim_in,
):
    base_vectors = jnp.concatenate((jnp.zeros((dim_in-1,)), jnp.ones((1,)), jnp.zeros((dim_in-1,))))
    get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (dim_in-k-1,), (dim_in,))
    def get_norm(i):
        e_i = get_vec(i)
        col_i = matrix_vector_product(e_i)
        return jnp.sum(col_i ** 2)
    return jax.lax.fori_loop(
        0, dim_in, 
        lambda i, temp : temp + get_norm(i), 
        0.
    )


def get_frobenius_norm_difference_sequential(
    J_vector_product,
    P_vector_product,
    dim_in,
):
    base_vectors = jnp.concatenate((jnp.zeros((dim_in-1,)), jnp.ones((1,)), jnp.zeros((dim_in-1,))))
    get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (dim_in-k-1,), (dim_in,))
    def get_norm(i):
        e_i = get_vec(i)

        col_i = J_vector_product(e_i)
        variance_I = jnp.sum(col_i ** 2)
        col_i = P_vector_product(col_i)
        variance_P = jnp.sum(col_i ** 2)
        variance = variance_I - variance_P
        return variance
    return jax.lax.fori_loop(
        0, dim_in, 
        lambda i, temp : temp + get_norm(i), 
        0.
    )