import jax.numpy as jnp

def ensemble_score_fun(
        model, 
        param_dicts_list
    ):

    def score_fun(batch):

        if not model.has_batch_stats:
            apply_model = lambda p_dict: model.apply_test(p_dict["params"], batch)
        else:
            apply_model = lambda p_dict: model.apply_test(p_dict["params"], p_dict["batch_stats"], batch)
            
        predictions = jnp.asarray([apply_model(param_dict) for param_dict in param_dicts_list])
        variance = jnp.var(predictions, axis=0) # variance over ensemble models
        return jnp.sum(variance, axis=1) # sum over output dimension

    return score_fun