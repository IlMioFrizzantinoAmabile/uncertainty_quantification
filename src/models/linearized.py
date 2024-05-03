import functools
import jax
from src.models.utils import has_batchstats

def get_linearized_model(model_flax, params_dict):

    if not has_batchstats(model_flax):
        def model(p, *, x):
            return model_flax.apply(p, x)
    else:
        batch_stats = params_dict['batch_stats']
        def model(p, *, x):
            return model_flax.apply(
                {'params': p, 'batch_stats': batch_stats}, 
                x,
                train=False,
                mutable=False
            )
    linearization_params = params_dict['params']

    @jax.jit
    def linear_model(params, x):
        model_on_x = functools.partial(model, x=x)
        delta_params = jax.tree_util.tree_map(lambda x, y: x-y, params, linearization_params)
        y, delta_y = jax.jvp(model_on_x, (linearization_params,), (delta_params,))
        return y + delta_y
    
    return linear_model