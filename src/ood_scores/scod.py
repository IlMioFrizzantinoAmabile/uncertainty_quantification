import jax
import jax.numpy as jnp
import numpy as np
import torch
import time
from src.models import compute_num_params
from src.datasets.utils import get_subset_loader, get_output_dim
from src.autodiff.hessian import get_sqrt_hessian_loss_explicit
from src.autodiff.jacobian import get_jacobian_vector_product, get_jacobianT_vector_product, get_jacobian_explicit
from src.estimators.frobenius import get_frobenius_norm
from src.sketches import SRFTSymSketch


def scod_score_fun(
        model, 
        params_dict, 
        train_loader, 
        args_dict, 
        use_eigenvals : bool = True, 
        gpu : bool = True
    ):
    #data_array = jnp.array([data[0] for data in train_loader.dataset])
    #data_array = jnp.asarray([train_loader.dataset[i][0] for i in range(int(0.9*args_dict["subsample_trainset"]))])
    #prior_scale = 1. / (2 * len(data_array) * args_dict['prior_std']**2) 
    #n_params = compute_num_params(params_dict["params"])
    trainset_size = int(0.9*args_dict["subsample_trainset"])
    train_loader = get_subset_loader(
            train_loader,
            trainset_size,
            batch_size = 1
        )
    prior_scale = 1. / (2 * trainset_size * args_dict['prior_std']**2) 
    n_params = compute_num_params(params_dict["params"])

    start = time.time()
    #sketch = SRFTSymSketch(n_params, data_array.shape[0], args_dict['n_eigenvec_hm'], gpu=gpu)
    sketch = SRFTSymSketch(n_params, trainset_size, args_dict['n_eigenvec_hm'], gpu=gpu)
    print(f"Initializing took {time.time()-start} seconds")
    output_dim = args_dict["output_dim"] #get_output_dim(args_dict["ID_dataset"])
    jacob_fun = get_jacobian_explicit(params_dict, model, output_dim=output_dim)
    hessian_loss_fun = get_sqrt_hessian_loss_explicit(params_dict, model, likelihood_type=args_dict["likelihood"], output_dim=output_dim)

    start = time.time()
    #for datapoint in data_array:
    for batch in train_loader:
        datapoint = jnp.array(batch[0][0].numpy())
        jac = jacob_fun(datapoint)
        hess = hessian_loss_fun(datapoint)
        hess_jac = hess @ jac
        #print(hess.shape, jac.shape, hess_jac.shape)
        hess_jac = torch.from_numpy(np.asarray(hess_jac))
        sketch.low_rank_update(0, hess_jac.T, 1.0)
    print(f"Updates took {time.time()-start} seconds")

    start = time.time()
    eigenval, eigenvec = sketch.get_range_basis()
    eigenval = jnp.asarray(eigenval[-args_dict['n_eigenvec_hm']:])
    eigenvec = jnp.asarray(eigenvec[:, -args_dict['n_eigenvec_hm']:])
    print(eigenvec.shape, eigenval.shape)
    print(f"Getting eigenvec/eigenval took {time.time()-start} seconds")
    print(f"returned {len(eigenval)} eigenvals  = {eigenval[:5]} ... {eigenval[-5:]}")


    # define the GGN vector product with the eigenvec decomposition, and its inverse and inverse sqrt
    @jax.jit
    def approx_ggn_vector_product(vector):
        return jnp.einsum("ab, b, cb, c-> a", eigenvec, eigenval, eigenvec, vector)
    if use_eigenvals:
        scale = jnp.sqrt(eigenval / (eigenval + prior_scale))
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return (vector @ eigenvec) * scale
    else:
        #@jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            print(vector.shape, eigenvec.shape)
            return vector @ eigenvec
    
    @jax.vmap
    @jax.jit
    def score_fun(datapoint):
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        inv_sqrt_fakeGGN_jacobian_vector_product = lambda vector: inv_sqrt_approx_ggn_vector_product(jacobianT_vector_product(vector))
        
        variance_I = get_frobenius_norm(
            jacobianT_vector_product,
            dim_in = args_dict["output_dim"],
        )
        variance_P = get_frobenius_norm(
            inv_sqrt_fakeGGN_jacobian_vector_product,
            dim_in = args_dict["output_dim"],
        )
        variance = variance_I - variance_P
        return variance * args_dict['prior_std']**2
    
    @jax.vmap
    @jax.jit
    def approx_ggn_quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        fake_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(approx_ggn_vector_product(jacobianT_vector_product(vector))))
        approx_qf = get_frobenius_norm(
            fake_quadratic_form,
            dim_in = args_dict["output_dim"],
        )
        return approx_qf
    
    return score_fun, eigenval, approx_ggn_quadratic_form