import jax
import jax.numpy as jnp
import functools
from jax import flatten_util
from src.models import compute_num_params
from src.datasets.utils import get_subset_loader
from src.autodiff.projection import get_projection_vector_product, get_loss_projection_vector_product
import time


def projected_ensemble_score_fun(
        model, 
        params_dict, 
        train_loader, 
        args_dict, 
    ):
    args_dict["likelihood"] = "regression"
    
    # define efficient GGN vector product
    trainset_size = int(0.9*args_dict["subsample_trainset"])
    
    # subsample train dataloader
    train_loader = get_subset_loader(
        train_loader,
        trainset_size,
        batch_size = args_dict["train_batch_size"]
    )

    if not args_dict["use_proj_loss"]:
        get_projection_vp = get_projection_vector_product
    else:
        get_projection_vp = get_loss_projection_vector_product

    projection_vector_product = get_projection_vp(
        params_dict,
        model,
        train_loader,
        likelihood_type = args_dict["likelihood"],
        num_epochs = args_dict["n_epochs_projected_ensemble"]
    )


    params = params_dict['params']
    n_params = compute_num_params(params)
    devectorize_fun = flatten_util.ravel_pytree(params)[1]


    #@jax.jit
    def get_sample(key):
        sample = jax.random.normal(key, shape=(n_params,))
        return projection_vector_product(sample)
    start = time.time()
    sample = get_sample(jax.random.PRNGKey(0))
    print(f"One PROJECTION sample took {time.time()-start} seconds")
    print("NORMmmm ", jnp.sum(sample**2))
    #start = time.time()
    #get_sample(jax.random.PRNGKey(1))
    #print(f"Again.. it took {time.time()-start} seconds")


    start = time.time()
    keys = jax.random.split(jax.random.PRNGKey(args_dict["model_seed"]), args_dict["ensemble_size"])

    samples = jax.vmap(get_sample)(keys)
    #samples = jax.lax.map(get_sample, keys)
    #samples = []
    #for key in keys:
    #    samples.append(get_sample(key))
    #    print("NORM ", jnp.sum(samples[-1]**2))
    #samples = jnp.asarray(samples)
    print(f"Projecting {args_dict['ensemble_size']} samples, dataset size {trainset_size}, with {n_params} params model -> took {time.time()-start:.3f} seconds")


    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_apply = lambda data, p: model.apply_test(p, attention_mask, relative_position_index, data)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_apply = lambda data, p: model.apply_test(p, batch_stats, data)
    else:
        model_apply = lambda data, p: model.apply_test(p, data)

    @jax.vmap
    @jax.jit
    def score_fun(datapoint):

        datapoint = jnp.expand_dims(datapoint, 0)
        model_on_data = functools.partial(model_apply, datapoint)

        def get_delta_prediction(sample):
            sample_tree = devectorize_fun(sample)
            _, delta_pred = jax.jvp(model_on_data, (params,), (sample_tree,))
            return delta_pred.reshape(-1)

        delta_predictions = jax.vmap(get_delta_prediction)(samples)

        variance = jnp.sum( delta_predictions ** 2 )
        return variance 

    
    return score_fun, None, None

