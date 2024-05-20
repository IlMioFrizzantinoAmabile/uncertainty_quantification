import jax
import jax.numpy as jnp
from src.models import compute_num_params
from src.autodiff.ggn import get_ggn_vector_product
from src.autodiff.hessian import get_hessian_vector_product
from src.autodiff.jacobian import get_jacobian_vector_product, get_jacobianT_vector_product
from src.lanczos.high_memory import high_memory_lanczos, precond_smart_lanczos
from src.estimators.frobenius import get_frobenius_norm
from src.sketches import No_sketch, Dense_sketch, SRFT_sketch
import time


def high_memory_lanczos_score_fun(
        model, 
        params_dict, 
        train_loader, 
        args_dict, 
        use_eigenvals : bool = True
    ):
    #data_array = jnp.array([data[0] for data in train_loader.dataset])
    data_array = jnp.asarray([train_loader.dataset[i][0] for i in range(int(0.9*args_dict["subsample_trainset"]))])
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
                data_array = (data_array, jnp.asarray([data[1] for data in train_loader.dataset])),
                likelihood_type = args_dict["likelihood"]
        )
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(0), shape=(n_params,)))
    print(f"One GGN vp took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(1), shape=(n_params,)))
    print(f"Again.. it took {time.time()-start} seconds")
        
    # perform Lanzos and find eigenval/eigenvec pairs
    start = time.time()
    key_lanczos = jax.random.PRNGKey(args_dict["lanczos_seed"])
    eigenvec, eigenval = high_memory_lanczos(key_lanczos, ggn_vector_product, n_params, args_dict["lanczos_hm_iter"])
    print(f"Lanczos {args_dict['lanczos_hm_iter']} iterations, dataset size {len(data_array)}, with {n_params} params model -> took {time.time()-start:.3f} seconds")
    print(f"returned {len(eigenval)} eigenvals = {eigenval[:5]} ... {eigenval[-5:]}")
    eigenvec = eigenvec[:, :args_dict['n_eigenvec_hm']]
    eigenval = eigenval[:args_dict['n_eigenvec_hm']]

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
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
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
    def approx_quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        fake_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(approx_ggn_vector_product(jacobianT_vector_product(vector))))
        approx_qf = get_frobenius_norm(
            fake_quadratic_form,
            dim_in = args_dict["output_dim"],
        )
        return approx_qf
    
    @jax.vmap
    @jax.jit
    def quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        real_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(ggn_vector_product(jacobianT_vector_product(vector))))
        qf = get_frobenius_norm(
            real_quadratic_form,
            dim_in = args_dict["output_dim"],
            sequential = True
        )
        return qf
    
    return score_fun, eigenval, approx_quadratic_form, quadratic_form



def smart_lanczos_score_fun(
        model, 
        params_dict, 
        train_loader, 
        args_dict, 
        use_eigenvals : bool = True
    ):
    #data_array = jnp.array([data[0] for data in train_loader.dataset])
    data_array = jnp.asarray([train_loader.dataset[i][0] for i in range(int(0.9*args_dict["subsample_trainset"]))])
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
                data_array = (data_array, jnp.asarray([data[1] for data in train_loader.dataset])),
                likelihood_type = args_dict["likelihood"]
        )
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(0), shape=(n_params,)))
    print(f"One GGN vp took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(1), shape=(n_params,)))
    print(f"Again.. it took {time.time()-start} seconds")

    # define the sketch operator (if needed)
    key_sketch = jax.random.PRNGKey(args_dict["sketch_seed"])
    if args_dict["sketch"] is None:
        sketch_op = No_sketch()
    elif args_dict["sketch"] == "srft":
        print(f"Use srft sketch with num params {n_params} and padding {args_dict['sketch_padding']} --> fake num params = {args_dict['sketch_padding']+n_params} must NOT have prime factors >127 (thanks JAX fft)")
        sketch_op = SRFT_sketch(key_sketch, n_params, args_dict['sketch_size'], padding=args_dict['sketch_padding'])
    elif args_dict["sketch"] == "dense":
        print(f"Use dense sketch with {'optimal' if args_dict['sketch_density'] is None else args_dict['sketch_density']} density")
        sketch_op = Dense_sketch(key_sketch, n_params, args_dict['sketch_size'], density=args_dict['sketch_density'])
    else:
        raise ValueError(f"Sketch '{args_dict['sketch']}' not supported. Use either 'srtf', 'dense' or None.")
        
    # perform Lanzos and find eigenval/eigenvec pairs
    start = time.time()
    key_lanczos = jax.random.PRNGKey(args_dict["lanczos_seed"])
    eigenvec, eigenval = precond_smart_lanczos(
        key_lanczos, 
        ggn_vector_product, 
        n_params, 
        args_dict["lanczos_lm_iter"],
        precondition_size = args_dict["n_eigenvec_hm"], 
        precondition_lanc_steps =  args_dict["lanczos_hm_iter"], 
        threshold = 0.5, 
        sketch_op = sketch_op
    )
    print(f"Lanczos {args_dict['lanczos_lm_iter']} iterations, dataset size {len(data_array)}, with {n_params} params model -> took {time.time()-start:.3f} seconds")
    print(f"returned {len(eigenval)} eigenvals = {eigenval[:5]} ... {eigenval[-5:]}")
    #eigenvec = eigenvec[:, :args_dict['n_eigenvec_lm']+args_dict['n_eigenvec_hm']]
    #eigenval = eigenval[:args_dict['n_eigenvec_lm']+args_dict['n_eigenvec_hm']]

    # define the GGN vector product with the eigenvec decomposition, and its inverse and inverse sqrt
    @jax.jit
    def approx_ggn_vector_product(vector):
        return sketch_op.T @ jnp.einsum("ab, b, cb, c-> a", eigenvec, eigenval, eigenvec, sketch_op @ vector) 

    if use_eigenvals:
        scale = jnp.sqrt(eigenval / (eigenval + prior_scale))
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return ((sketch_op @ vector).T @ eigenvec) * scale
    else:
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return (sketch_op @ vector).T @ eigenvec
        
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
    def quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        real_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(ggn_vector_product(jacobianT_vector_product(vector))))
        qf = get_frobenius_norm(
            real_quadratic_form,
            dim_in = args_dict["output_dim"],
            sequential = True
        )
        return qf
    
    @jax.vmap
    @jax.jit
    def approx_quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        fake_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(approx_ggn_vector_product(jacobianT_vector_product(vector))))
        approx_qf = get_frobenius_norm(
            fake_quadratic_form,
            dim_in = args_dict["output_dim"],
        )
        return approx_qf
    
    return score_fun, eigenval, approx_quadratic_form, quadratic_form