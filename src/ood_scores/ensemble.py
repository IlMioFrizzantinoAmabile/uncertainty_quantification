import jax.numpy as jnp
from src.models.utils import has_batchstats
import jax 

def ensemble_score_fun(
        model, 
        param_dicts_list
    ):

    def score_fun(batch):

        if not has_batchstats(model):
            apply_model = lambda p_dict: model.apply(p_dict["params"], batch)
        else:
            apply_model = lambda p_dict: model.apply(
                {
                    "params" : p_dict["params"],
                    "batch_stats" : p_dict["batch_stats"]
                }, 
                batch, train=False, mutable=False)
            #apply_model = lambda p_dict: model.apply(
            #    p_dict,
            #    batch, train=False, mutable=False)
            
        #predictions = list(map(apply_model, param_dicts_list))
        #predictions = jnp.asarray(predictions)
        predictions = jnp.asarray([apply_model(param_dict) for param_dict in param_dicts_list])
        variance = jnp.var(predictions, axis=0) # variance over ensemble models
        return jnp.sum(variance, axis=1) # sum over output dimension

    return score_fun