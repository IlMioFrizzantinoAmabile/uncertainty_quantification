import jax
import jax.numpy as jnp
import scipy
from functools import partial
from matfree import decomp

######################
# low memory lanczos #
# performed without reothogonalization.

def low_memory_lanczos(
        key: jax.random.PRNGKey, 
        mv_prod, 
        dim: int, 
        n_iter: int, 
        sketch_op = None
    ):
    key, key_lancz = jax.random.split(key)
    if sketch_op is None:
        skvs, alphas, betas = low_memory_lanczos_to_tridiag(key_lancz, mv_prod, dim, n_iter)
    else:
        skvs, alphas, betas = low_memory_lanczos_to_tridiag_sketch(key_lancz, mv_prod, dim, n_iter, sketch_op)
        
    eig_val, eig_vec = scipy.linalg.eigh_tridiagonal(jnp.array(alphas), jnp.array(betas), lapack_driver='stebz')
    eig_vec, eig_val = jnp.array(eig_vec), jnp.array(eig_val)
    sketched_eig_vec = skvs.T @ eig_vec
    # flip eigenvalues and eigenvectors so that they are in decreasing order
    eig_val = jnp.flip(eig_val)
    sketched_eig_vec = jnp.flip(sketched_eig_vec, axis=1)
    return sketched_eig_vec, eig_val


#@partial(jax.jit, static_argnames=['mv_prod', 'dim', 'n_iter'])
def low_memory_lanczos_to_tridiag(key, mv_prod, dim, n_iter):
    v_old = jax.random.normal(key, shape=(dim, ))
    #v_old /= jnp.sqrt(dim)
    v_old /= jax.numpy.sqrt(v_old.T @ v_old)

    skvs = jnp.zeros((n_iter, v_old.shape[0]))
    skvs = skvs.at[0].set(v_old)
    alphas = jnp.zeros(n_iter)
    betas = jnp.zeros(n_iter - 1)

    wp = mv_prod(v_old)
    a = jnp.dot(wp, v_old)
    w = wp - a * v_old
    alphas = alphas.at[0].set(a)

    @jax.jit
    def lanczos_step(i, state):
        (w, betas, skvs, alphas, v_old) = state
        b = jnp.linalg.norm(w)
        betas = betas.at[i - 1].set(b)
        v_new = w / b
        skvs = skvs.at[i].set(v_new)
        wp = mv_prod(v_new)
        a = jnp.dot(wp, v_new)
        alphas = alphas.at[i].set(a)
        w = wp - a * v_new - b * v_old
        v_old = v_new
        return (w, betas, skvs, alphas, v_old)

    state = (w, betas, skvs, alphas, v_old)
    state = jax.lax.fori_loop(1, n_iter, lanczos_step, state)
    (w, betas, skvs, alphas, v_old) = state

    return skvs, alphas, betas


#@partial(jax.jit, static_argnames=['mv_prod', 'dim', 'n_iter', 'sketch_op'])
def low_memory_lanczos_to_tridiag_sketch(key, mv_prod, dim, n_iter, sketch_op):
    v_old = jax.random.normal(key, shape=(dim, ))
    #v_old /= jnp.sqrt(dim)
    v_old /= jax.numpy.sqrt(v_old.T @ v_old)

    skv_old = sketch_op @ v_old
    skvs = jnp.zeros((n_iter, skv_old.shape[0]))
    skvs = skvs.at[0].set(skv_old)
    alphas = jnp.zeros(n_iter)
    betas = jnp.zeros(n_iter - 1)

    wp = mv_prod(v_old)
    a = jnp.dot(wp, v_old)
    w = wp - a * v_old
    alphas = alphas.at[0].set(a)

    #jax.jit
    def lanczos_step(i, state):
        (w, betas, skvs, alphas, v_old) = state
        b = jnp.linalg.norm(w)
        betas = betas.at[i - 1].set(b)
        v_new = w / b
        skvs = skvs.at[i].set(sketch_op @ v_new)
        wp = mv_prod(v_new)
        a = jnp.dot(wp, v_new)
        alphas = alphas.at[i].set(a)
        w = wp - a * v_new - b * v_old
        v_old = v_new
        return (w, betas, skvs, alphas, v_old)

    state = (w, betas, skvs, alphas, v_old)
    state = jax.lax.fori_loop(1, n_iter, lanczos_step, state)
    (w, betas, skvs, alphas, v_old) = state

    return skvs, alphas, betas



############################
# low memory smart lanczos #
# performed without reothogonalization. Sketched scalar product are stored in order to retrieve approximate eigenvalues
def smart_lanczos(key, mv_prod, dim, n_iter, threshold=0.5, sketch_op=None):
    skvs, M = smart_lanczos_to_tridiag_sketch(key, mv_prod, dim, n_iter, sketch_op)
    U, D, Y = jnp.linalg.svd(skvs.T, full_matrices=False)
    X = (1 / D)[:, None] * Y @ M @ Y.T * (1 / D)
    k = jnp.sum(D > threshold)
    X = X[:k, :k]
    eigval, eigvec = jnp.linalg.eigh(X)
    eigvec = U[:, :k] @ eigvec
    
    sort_order = jnp.flip(jnp.argsort(eigval))
    eigvec = eigvec[:, sort_order]
    eigval = eigval[sort_order]
    return eigvec, eigval


#@partial(jax.jit, static_argnames=['mv_prod', 'dim', 'n_iter', 'sketch_op'])
def smart_lanczos_to_tridiag_sketch(key, mv_prod, dim, n_iter, sketch_op):
    v_old = jax.random.normal(key, shape=(dim, ))
    #v_old /= jnp.sqrt(dim)
    v_old /= jax.numpy.sqrt(v_old.T @ v_old)
    skv_old = sketch_op @ v_old

    skvs = jnp.zeros((n_iter, skv_old.shape[0]))
    skvs = skvs.at[0].set(skv_old)
    M = jnp.zeros((n_iter, n_iter))

    wp = mv_prod(v_old)
    a = jnp.dot(wp, v_old)
    w = wp - a * v_old
    M = M.at[0, 0].set(a)

    @jax.jit
    def lanczos_step(i, state):
        (w, M, skvs, v_old) = state
        b = jnp.linalg.norm(w)
        v_new = w / b
        wp = mv_prod(v_new)
        M = M.at[i].set(skvs @ (sketch_op @ wp))
        M = M.at[:, i].set(M[i])
        skvs = skvs.at[i].set(sketch_op @ v_new)
        a = jnp.dot(wp, v_new)
        w = wp - a * v_new - b * v_old
        v_old = v_new
        M = M.at[i, i].set(a)
        M = M.at[i, i - 1].set(b)
        M = M.at[i - 1, i].set(b)
        return (w, M, skvs, v_old)

    state = (w, M, skvs, v_old)
    state = jax.lax.fori_loop(1, n_iter, lanczos_step, state)
    (w, M, skvs, v_old) = state

    return skvs, M
