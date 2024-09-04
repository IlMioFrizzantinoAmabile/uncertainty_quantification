import functools
import jax

def get_linearized_model(model, params_dict):

    if not model.has_batch_stats:
        def model_apply(p, *, x):
            return model.apply_test(p, x)
    else:
        batch_stats = params_dict['batch_stats']
        def model_apply(p, *, x):
            return model.apply_test(p, batch_stats, x)
    linearization_params = params_dict['params']

    @jax.jit
    def linear_model(params, x):
        model_on_x = functools.partial(model_apply, x=x)
        delta_params = jax.tree_util.tree_map(lambda x, y: x-y, params, linearization_params)
        y, delta_y = jax.jvp(model_on_x, (linearization_params,), (delta_params,))
        return y + delta_y
    
    return linear_model