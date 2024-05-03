import jax
import jax.numpy as jnp
from src.models import compute_num_params
from src.autodiff.ggn import get_ggn_vector_product
from src.autodiff.hessian import get_hessian_vector_product
from src.autodiff.jacobian import get_jacobian_vector_product, get_jacobianT_vector_product
from src.estimators.hutchinson import get_diagonal
from src.estimators.frobenius import get_frobenius_norm
import time


def diagonal_lla_score_fun(model, params_dict, train_loader, args_dict):
    #data_array = jnp.array([data[0] for data in train_loader.dataset])
    data_array = jnp.array([train_loader.dataset[i][0] for i in range(int(0.9*args_dict["subsample_trainset"]))])
    prior_scale = 1. / (2 * len(data_array) * args_dict['prior_std']**2) 
    n_params = compute_num_params(params_dict["params"])

    # define efficient GGN vector product
    if not args_dict["use_hessian"]:
        ggn_vector_product = get_ggn_vector_product(
                params_dict,
                model,
                data_array = data_array,
                likelihood_type = args_dict["likelihood"]
        )
    else:
        print("Using the Hessian instead of the GGN")
        ggn_vector_product = get_hessian_vector_product(
                params_dict,
                model,
                data_array = (data_array, jnp.array([data[1] for data in train_loader.dataset])),
                likelihood_type = args_dict["likelihood"]
        )
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(0), shape=(n_params,)))
    print(f"One GGN vp took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(1), shape=(n_params,)))
    print(f"Again.. it took {time.time()-start} seconds")
    
    # perform hutchinson and estimate the diagonal
    start = time.time()
    ggn_diagonal = get_diagonal(
        ggn_vector_product,
        n_params,
        sequential = True,
        n_samples = args_dict["hutchinson_samples"],
        key = jax.random.PRNGKey(args_dict["hutchinson_seed"]),
        estimator = "Rademacher"
    ) # for reference: MNIST full dataset on Lenet takes around 7 second every 100 samples (vector products) -> exact diagonal ggn takes around 1 hour
    print(f"ggn diagonal, dataset size {len(data_array)}, with {n_params} params model -> took {time.time()-start:.3f} seconds")
    min_value = jnp.min(ggn_diagonal)
    if min_value < 0:
        ggn_diagonal -= jnp.min(ggn_diagonal)
        ggn_diagonal += 1e-3
        print(f"Min is negative :( - {min_value}")

    # define the GGN vector product with the diagnal approx
    @jax.jit
    def approx_ggn_vector_product(vector):
        return jnp.einsum("a, a -> a", ggn_diagonal, vector)
    
    parameter_std_diagonal = jnp.sqrt(args_dict['prior_std'] / (ggn_diagonal + prior_scale))

    @jax.vmap
    @jax.jit
    def score_fun(datapoint):
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        std_jacobian_vector_product = lambda vector: parameter_std_diagonal * jacobianT_vector_product(vector)
        
        variance = get_frobenius_norm(
            std_jacobian_vector_product,
            dim_in = args_dict["output_dim"],
        )
        return variance
    
    @jax.vmap
    @jax.jit
    def ggn_quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        real_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(ggn_vector_product(jacobianT_vector_product(vector))))
        qf_true = get_frobenius_norm(
            real_quadratic_form,
            dim_in = args_dict["output_dim"],
            sequential = True
        )
        return qf_true
    
    @jax.vmap
    @jax.jit
    def approx_ggn_quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        fake_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(approx_ggn_vector_product(jacobianT_vector_product(vector))))
        qf_fake = get_frobenius_norm(
            fake_quadratic_form,
            dim_in = args_dict["output_dim"],
        )
        return qf_fake
    
    return score_fun, ggn_quadratic_form, approx_ggn_quadratic_form