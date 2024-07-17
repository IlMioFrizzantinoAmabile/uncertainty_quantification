import jax
import jax.numpy as jnp
from src.models import compute_num_params
from src.datasets.utils import get_subset_loader
from src.autodiff.ggn import get_ggn_vector_product, get_ggn_vector_product_dataloader
from src.autodiff.hessian import get_hessian_vector_product, get_hessian_vector_product_dataloader
from src.autodiff.jacobian import get_jacobian_vector_product, get_jacobianT_vector_product
from src.estimators.hutchinson import get_diagonal
from src.estimators.frobenius import get_frobenius_norm
import time


def diagonal_lla_score_fun(model, params_dict, train_loader, args_dict):
    # define efficient GGN vector product
    trainset_size = int(0.9*args_dict["subsample_trainset"])
    if not args_dict["serialize_ggn_on_batches"]:
        # subsample train dataset
        data_array = jnp.asarray([train_loader.dataset[i][0] for i in range(trainset_size)])
        # get matrix vector product fun
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
                    data_array = (data_array, jnp.asarray([data[1] for data in train_loader.dataset])),
                    likelihood_type = args_dict["likelihood"]
            )
    else:
        # subsample train dataloader
        train_loader = get_subset_loader(
            train_loader,
            trainset_size,
            batch_size = args_dict["train_batch_size"]
        )
        # get matrix vector product fun
        if not args_dict["use_hessian"]:
            ggn_vector_product = get_ggn_vector_product_dataloader(
                params_dict,
                model,
                train_loader,
                likelihood_type = args_dict["likelihood"]
            )
        else:
            print("Using the Hessian instead of the GGN")
            ggn_vector_product = get_hessian_vector_product_dataloader(
                params_dict,
                model,
                train_loader,
                likelihood_type = args_dict["likelihood"]
            )
    prior_scale = 1. / (2 * trainset_size * args_dict['prior_std']**2) 
    n_params = compute_num_params(params_dict["params"])
    
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
    print(f"ggn diagonal, dataset size {trainset_size}, with {n_params} params model -> took {time.time()-start:.3f} seconds")
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