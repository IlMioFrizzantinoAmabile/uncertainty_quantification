import jax
import jax.numpy as jnp

def max_logit_score_fun(
        model, 
        params_dict
    ):

    if model.has_attentionmask:
        model_on_params = lambda batch: model.apply_test(params_dict['params'], params_dict['attention_mask'], params_dict['relative_position_index'], batch)
    elif model.has_batch_stats:
        model_on_params = lambda batch: model.apply_test(params_dict['params'], params_dict['batch_stats'], batch)
    else:
        model_on_params = lambda batch: model.apply_test(params_dict['params'], batch)
    
    @jax.jit
    def score_fun(batch):

        predictions = model_on_params(batch)
        max_logits = jnp.max(predictions, axis=-1)
        return -max_logits

    return score_fun