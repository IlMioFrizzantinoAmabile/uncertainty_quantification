import jax
from jax import numpy as jnp
from functools import partial

from src.estimators.hutchinson import get_trace

def logdeterminant_bound(N, mu_1, mu_2, a):
    """
    Returns upper or lower bounds on the log-determinant of a matrix S
    :param N:
        size of the matrix S
    :type N:
        int
    :param mu_1 (trS):
        estimate for the trace of S
    :type trS:
        double
    :param mu_2 (trSS):
        estimate for the trace of SS
    :type trSS:
        double
    :param a:
        lower or upper bound on the eigenvalue of S, depending on if you want a lower bound on logdet or upper bound.
    :type a:
        double
    :return:
        lower or upper bound
    :rtype:
        double
    """
    sub_num = (a * mu_1 - mu_2)
    sub_den = (a * N - mu_1)
    t_ = sub_num / sub_den
    log_t_ = jnp.log(sub_num) - jnp.log(sub_den)
    S = mu_1 * t_**2 - t_*mu_1**2
    term_1 = jnp.log(a)*(t_**2 * mu_1 - t_*mu_2)
    term_2 = log_t_*(mu_1*mu_2 - mu_1**3)
    rr = (term_1 + term_2)/S
    return rr

#@partial(jax.jit, static_argnames=['dim', 'sequential', 'n_samples', 'key', 'estimator'])
def get_logdeterminant(
    matrix_vector_product,
    dim,
    sequential = True,
    n_samples: int = None,
    key: jax.random.PRNGKey = None,
    estimator = "Rademacher",
):
    
    trace = get_trace(
        matrix_vector_product,
        dim,
        sequential = sequential,
        n_samples = n_samples,
        key = key,
        estimator = estimator)

    matrix_square_vector_product = lambda v: matrix_vector_product(matrix_vector_product(v))
    trace_square = get_trace(
        matrix_square_vector_product,
        dim,
        sequential = sequential,
        n_samples = n_samples,
        key = key,
        estimator = estimator)
    
    logdeterminant = logdeterminant_bound(dim, trace, trace_square, trace)

    #jax.debug.print("trace {trace}, trace_sq {trace_sq}, log_det {log_det}", trace=trace, trace_sq=trace_square, log_det=logdeterminant)
    return logdeterminant