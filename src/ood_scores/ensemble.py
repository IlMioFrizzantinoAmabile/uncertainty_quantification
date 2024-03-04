import jax.numpy as jnp
from src.models.utils import has_batchstats

def ensemble_score_fun(
        model, 
        param_dicts_list
    ):

    def score_fun(batch):

        if not has_batchstats(model):
            apply_model = lambda p_dict: model.apply(p_dict["params"], batch)
        else:
            apply_model = lambda p_dict: model.apply(p_dict, batch, train=False, mutable=False)
            
        predictions = list(map(apply_model, param_dicts_list))
        predictions = jnp.array(predictions)
        variance = jnp.var(predictions, axis=0) # variance over ensemble models
        return jnp.sum(variance, axis=1) # sum over output dimension

    return score_fun