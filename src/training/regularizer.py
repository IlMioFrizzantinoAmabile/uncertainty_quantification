import jax
from typing import Literal

from src.models import compute_num_params
from src.autodiff.ggn import get_ggn_vector_product
from src.autodiff.ntk import get_ntk_vector_product
from src.estimators.determinant import get_logdeterminant

import time


def log_determinant_ntk(
    model, 
    params_dict, 
    x, 
    dim: int,
    prior_precision: float = 1.,
    lik_precision: float = 1.,
    likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification",
    sequential = True,
    n_samples: int = None,
    key: jax.random.PRNGKey = None,
    estimator = "Rademacher"
):

    ntk_vector_product = get_ntk_vector_product(
        params_dict,
        model,
        x,
        single_datapoint = False,
        likelihood_type = likelihood
    )
    ntk_plus_prior_vector_product = lambda v: (1./prior_precision) * ntk_vector_product(v) + (1./lik_precision) * v
    jit_vp = jax.jit(ntk_plus_prior_vector_product)

    start = time.time()
    jit_vp(jax.random.normal(jax.random.PRNGKey(0), shape=(dim,)))
    print(f"One NTK vp took {time.time()-start} seconds")
    start = time.time()
    jit_vp(jax.random.normal(jax.random.PRNGKey(1), shape=(dim,)))
    print(f"Again.. it took {time.time()-start} seconds")

    logdeterminant = get_logdeterminant(
        ntk_plus_prior_vector_product,
        dim,
        sequential = sequential,
        n_samples = n_samples,
        key = key,
        estimator = estimator,
    )

    return logdeterminant


def log_determinant_ggn(
    model, 
    params_dict, 
    x, 
    dim: int,
    prior_precision: float = 1.,
    lik_precision: float = 1.,
    likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification",
    sequential = True,
    n_samples: int = None,
    key: jax.random.PRNGKey = None,
    estimator = "Rademacher"
):

    ggn_vector_product = get_ggn_vector_product(
        params_dict,
        model,
        x,
        single_datapoint = False,
        likelihood_type = likelihood
    )
    ggn_plus_prior_vector_product = lambda v: lik_precision * ggn_vector_product(v) + prior_precision * v
    jit_vp = jax.jit(ggn_plus_prior_vector_product)

    start = time.time()
    jit_vp(jax.random.normal(jax.random.PRNGKey(0), shape=(dim,)))
    print(f"One GGN vp took {time.time()-start} seconds")
    start = time.time()
    jit_vp(jax.random.normal(jax.random.PRNGKey(1), shape=(dim,)))
    print(f"Again.. it took {time.time()-start} seconds")


    #jax.debug.print("run logdet with ggn vp {x}", x=time.time()-start)
    logdeterminant = get_logdeterminant(
        jit_vp,
        dim,
        sequential = sequential,
        n_samples = n_samples,
        key = key,
        estimator = estimator,
    )

    return logdeterminant