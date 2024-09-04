import jax
import jax.numpy as jnp
import scipy
import numpy as np
from functools import partial
from matfree.decomp import tridiag_sym #_tridiag_reortho_full #lanczos_full_reortho #lanczos_tridiag_full_reortho
from matfree import decomp
from src.lanczos.low_memory import smart_lanczos


#######################
# high-memory Lanczos #
# implementation from matfree library: it performs full reorthogonalization 
def high_memory_lanczos(
        key: jax.random.PRNGKey, 
        mv_prod, 
        dim: int, 
        n_iter: int):
    key, key_lancz = jax.random.split(key)
    v0 = jax.random.normal(key_lancz, shape=(dim, ))
    #v0 /= jnp.sqrt(dim)
    v0 /= jnp.sqrt(v0.T @ v0)
    #lanczos_alg = lanczos_full_reortho(n_iter - 1)
    #basis, (diag, offdiag) = decomp.decompose_fori_loop(v0, mv_prod, algorithm=lanczos_alg)
    #estimate = tridiag_sym(n_iter, materialize=False)
    estimate = tridiag_sym(n_iter, materialize=True)
    decomposition, remainder = estimate(mv_prod, v0)
    #basis, (diag, offdiag) = decomposition
    basis, matrix = decomposition
    #hm_eig_val, hm_trid_eig_vec = jax.scipy.linalg.eigh_tridiagonal(diag, offdiag)#, lapack_driver='stebz')
    hm_eig_val, hm_trid_eig_vec = jax.scipy.linalg.eigh(matrix)
    # flip eigenvalues and eigenvectors so that they are in decreasing order
    hm_trid_eig_vec = jnp.stack(list(hm_trid_eig_vec.T)[::-1], axis=1)
    hm_eig_val = jnp.array(list(hm_eig_val)[::-1])
    # multiply eigenvector matrices
    hm_eig_vec = basis.T @ hm_trid_eig_vec
    
    return hm_eig_vec, hm_eig_val 



#############################################################
# low memory smart lanczos with high memory precontitioning #

def precond_smart_lanczos(
        key, 
        mv_prod, 
        dim: int, 
        n_iter: int, 
        precondition_size: int = 3, 
        precondition_lanc_steps: int = None, 
        threshold: float = 0.5, 
        sketch_op = None
    ):
    if precondition_size != 0:
        key, key_prec = jax.random.split(key)
        mv_prod, prec_vec, prec_val = precondition(key_prec, mv_prod, dim, precondition_size, lanc_steps=precondition_lanc_steps)
        prec_vec = sketch_op @ prec_vec
        key, key_smart = jax.random.split(key)
        eigvec, eigval = smart_lanczos(key_smart, mv_prod, dim, n_iter, threshold=threshold, sketch_op=sketch_op)
        eigvec = jnp.concatenate([prec_vec, eigvec], axis=1)
        eigval = jnp.concatenate([prec_val, eigval])
        # make sure they are sorted by decreasing eigval
        sort_order = jnp.flip(jnp.argsort(eigval))
        eigvec = eigvec[:, sort_order]
        eigval = eigval[sort_order]
        return eigvec, eigval
    else:
        key, key_smart = jax.random.split(key)
        return smart_lanczos(key_smart, mv_prod, dim, n_iter, threshold=threshold, sketch_op=sketch_op)


def precondition(
        key, 
        mv_prod, 
        dim: int, 
        precondition_size : int, 
        lanc_steps: int = None
    ):
    if lanc_steps is None:
        lanc_steps = 2 * precondition_size
    eigvec, eigval = high_memory_lanczos(key, mv_prod, dim, lanc_steps)
    eigvec = eigvec[:, :precondition_size]
    eigval = eigval[:precondition_size]

    @jax.jit
    def precond_mv(v):
        v -= eigvec @ (eigvec.T @ v)
        v = mv_prod(v)
        v -= eigvec @ (eigvec.T @ v)
        return v
    
    return precond_mv, eigvec, eigval