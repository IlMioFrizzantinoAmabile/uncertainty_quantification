import jax
import jax.numpy as jnp
from jax import flatten_util
import optax
import tqdm
import time
from src.models import compute_num_params
from src.training.losses import get_likelihood


def swag_score_fun(
        model, 
        params_dict, 
        train_loader, 
        args_dict, 
        diag_only=True, max_num_models=20, swa_c_epochs=1, swa_c_batches=None,
        swa_lr=0.01, momentum=0.9, wd=3e-4
    ):

    ########################
    # initialize optimizer #
    optimizer = optax.chain(
        optax.add_decayed_weights(wd),
        optax.sgd(swa_lr, momentum=momentum)
    )
    opt_state = optimizer.init(params_dict['params'])

    
    #############
    # init loss #
    negative_log_likelihood, extra_stats_function = get_likelihood(
        likelihood = args_dict["likelihood"],
        class_frequencies = train_loader.dataset.dataset.class_frequencies if args_dict["likelihood"]=="binary_multiclassification" else None
    )
    if not model.has_batch_stats:
        if not model.has_attentionmask:
            # MLP, LeNet, ConvNeXt
            @jax.jit
            def loss_function_train(params, x, y):
                preds = model.apply_train(params, x)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, )
        else:
            # SWIN
            @jax.jit
            def loss_function_train(params, attention_mask, relative_position_index, x, y, key_dropout):
                preds, new_model_state = model.apply_train(params, attention_mask, relative_position_index, x, key_dropout)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, new_model_state)
    elif model.has_batch_stats:
        if not model.has_dropout:
            # ResNet, GoogleNet
            @jax.jit
            def loss_function_train(params, batch_stats, x, y):
                preds, new_model_state = model.apply_train(params, batch_stats, x)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, new_model_state)
        else:
            # VAN
            @jax.jit
            def loss_function_train(params, batch_stats, x, y, key_dropout):
                preds, new_model_state = model.apply_train(params, batch_stats, x, key_dropout)
                loss = negative_log_likelihood(preds, y)
                acc_or_sse = extra_stats_function(preds, y)
                return loss, (acc_or_sse, new_model_state)


    if not model.has_batch_stats:
        if not model.has_attentionmask:
            # MLP, LeNet
            @jax.jit
            def train_step(opt_state, params_dict, x, y):
                loss_fn = lambda p: loss_function_train(p, x, y)
                # Get loss, gradients for loss, and other outputs of loss function
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, ) = ret
                # Update parameters
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
                return opt_state, params_dict, loss, acc_or_sse
        else:
            # SWIN
            @jax.jit
            def train_step(opt_state, params_dict, x, y, key_dropout):
                loss_fn = lambda p: loss_function_train(p, params_dict['attention_mask'], params_dict['relative_position_index'], x, y, key_dropout)
                # Get loss, gradients for loss, and other outputs of loss function
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, new_model_state) = ret
                # Update parameters and batch statistics
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict = {
                    'params' : optax.apply_updates(params_dict['params'], param_updates),
                    'attention_mask' : new_model_state['attention_mask'],
                    'relative_position_index' : new_model_state['relative_position_index']
                }
                return opt_state, params_dict, loss, acc_or_sse
            key_dropout = jax.random.PRNGKey(420)
    elif model.has_batch_stats:
        if not model.has_dropout:
            # ResNet, GoogleNet
            @jax.jit
            def train_step(opt_state, params_dict, x, y):
                loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y)
                # Get loss, gradients for loss, and other outputs of loss function
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, new_model_state) = ret
                # Update parameters and batch statistics
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict = {
                    'params' : optax.apply_updates(params_dict['params'], param_updates),
                    'batch_stats' : new_model_state['batch_stats']
                }
                return opt_state, params_dict, loss, acc_or_sse
        else:
            # VAN
            @jax.jit
            def train_step(opt_state, params_dict, x, y, key_dropout):
                loss_fn = lambda p: loss_function_train(p, params_dict['batch_stats'], x, y, key_dropout)
                # Get loss, gradients for loss, and other outputs of loss function
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
                loss, (acc_or_sse, new_model_state) = ret
                # Update parameters and batch statistics
                param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
                params_dict = {
                    'params' : optax.apply_updates(params_dict['params'], param_updates),
                    'batch_stats' : new_model_state['batch_stats']
                }
                return opt_state, params_dict, loss, acc_or_sse
            key_dropout = jax.random.PRNGKey(420)
    
    def fit_batch_stats(train_loader, params_dict):
        for batch in train_loader:
            X = jnp.asarray(batch[0].numpy())
            _, new_model_state = model.apply(
                params_dict,
                X,
                train = True,
                mutable = ['batch_stats'])
            params_dict = {
                'params' : params_dict['params'],
                'batch_stats' : new_model_state['batch_stats']
            }
        return params_dict['batch_stats']

        
    ######################
    # define SWAG update #
    #@jax.jit
    def collect_model(params_dict, mean, sq_mean, n_models, cov_mat_sqrt):

        params = flatten_util.ravel_pytree(params_dict['params'])[0]

        # first moment
        mean = mean * n_models / (
            n_models + 1.0
        ) + params / (n_models + 1.0)

        # second moment
        sq_mean = sq_mean * n_models / (
            n_models + 1.0
        ) + params ** 2 / (n_models + 1.0)

        # square root of covariance matrix
        if diag_only is False:

            # block covariance matrices, store deviation from current mean
            dev = (params - mean)

            cov_mat_sqrt.append(dev)

            # remove first column if we have stored too many models
            if n_models + 1 > max_num_models:
                cov_mat_sqrt = cov_mat_sqrt[1:]

        n_models += 1
        return mean, sq_mean, cov_mat_sqrt, n_models
        
    ###################################
    # start stochastic gradient steps #
    num_params = compute_num_params(params_dict['params'])
    mean = jnp.zeros(num_params)
    sq_mean = jnp.zeros(num_params)
    cov_mat_sqrt = []
    n_models = 0

    if swa_c_epochs is not None and swa_c_batches is not None:
        raise RuntimeError("One of swa_c_epochs or swa_c_batches must be None!")
    if swa_c_epochs is not None:
        n_epochs = swa_c_epochs * max_num_models
    else:
        n_epochs = 1 + (max_num_models * swa_c_batches) // len(train_loader)


    start = time.time()
    #epochs_bar = tqdm.tqdm(range(n_epochs))
    #for epoch in epochs_bar:
    for epoch in range(n_epochs):
        loss_avg = 0.
        batches_bar = tqdm.tqdm(enumerate(train_loader))
        for batch_idx, batch in batches_bar:
        #for batch_idx, batch in enumerate(train_loader):
            X = jnp.asarray(batch[0].numpy())
            Y = jnp.asarray(batch[1].numpy())


            if model.has_dropout:
                k, key_dropout = jax.random.split(key_dropout, 2)
                opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y, k)
            else:
                opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y)

            loss_avg += batch_loss.item()

            if swa_c_batches is not None and (epoch*len(train_loader) + batch_idx + 1) % swa_c_batches == 0:
                mean, sq_mean, cov_mat_sqrt, n_models = collect_model(params_dict, mean, sq_mean, n_models, cov_mat_sqrt)
                if n_models == max_num_models:
                    break
            batches_bar.set_description(f"Epoch {epoch}/{n_epochs}, batch {batch_idx}/{len(train_loader)}, loss = {loss_avg/len(train_loader):.4f}")

        if swa_c_epochs is not None and epoch % swa_c_epochs == 0:
            mean, sq_mean, cov_mat_sqrt, n_models = collect_model(params_dict, mean, sq_mean, n_models, cov_mat_sqrt)
        
        #epochs_bar.set_description(f"Epoch {epoch}/{n_epochs}, loss = {loss_avg/len(train_loader):.4f}")
    
    print(f"Models collection took {time.time()-start} seconds")

    cov_mat_sqrt = jnp.asarray(cov_mat_sqrt)
    cov_mat_sqrt = cov_mat_sqrt.transpose()
    print("Covariance matrix done, shape", cov_mat_sqrt.shape)

    ##########################
    # define sample function #
    @jax.jit
    def sample(key, scale=0.5):
        key1, key2 = jax.random.split(key, 2)

        # draw diagonal variance sample
        var = jnp.clip(sq_mean - mean ** 2, 1e-30, None)
        rand_sample = (var ** 0.5) * jax.random.normal(key1, shape=(num_params,))

        # if covariance draw low rank sample
        if diag_only is False:
            cov_sample = cov_mat_sqrt @ jax.random.normal(key2, shape=(max_num_models,))
            cov_sample /= (max_num_models - 1) ** 0.5
            rand_sample += cov_sample

        # update sample with mean and scale
        sample = mean + scale**0.5 * rand_sample

        return sample
    

    output_dim = Y.shape[-1]
    devectorize_fun = flatten_util.ravel_pytree(params_dict['params'])[1]

    #@jax.jit
    #def score_fun(datapoints):
    #    key = jax.random.PRNGKey(0)
    #    preds = []
    #    for s in range(50):
    #        key, key_s = jax.random.split(key, 2)
    #        sample_params = devectorize_fun(sample(key_s, scale=0.5))
    #        
    #        if model.has_batch_stats:
    #            #batch_stats = fit_batch_stats(train_loader, params_dict)
    #            pred = model.apply_test(sample_params, params_dict["batch_stats"], datapoints)
    #        elif model.has_attentionmask:
    #            pred = model.apply_test(sample_params, params_dict["attention_mask"], params_dict["relative_position_index"], datapoints)
    #        else:
    #            pred = model.apply_test(sample_params, datapoints)
    #        preds.append(pred)
    #    preds = jnp.asarray(preds)
    #    return preds.var(axis=0).sum(axis=-1)

    if model.has_batch_stats:
        model_apply = lambda sample_params, datapoints: model.apply_test(sample_params, params_dict["batch_stats"], datapoints)
    elif model.has_attentionmask:
        model_apply = lambda sample_params, datapoints: model.apply_test(sample_params, params_dict["attention_mask"], params_dict["relative_position_index"], datapoints)
    else:
        model_apply = lambda sample_params, datapoints: model.apply_test(sample_params, datapoints)

    @jax.jit
    def score_fun(datapoints):
    
        def scan_body(carry, key):
            sample_params = devectorize_fun(sample(key, scale=0.5))
            pred = model_apply(sample_params, datapoints)
            return carry, pred

        keys = jax.random.split(jax.random.PRNGKey(0), 50)

        _, preds = jax.lax.scan(scan_body, None, keys)

        return preds.var(axis=0).sum(axis=-1)


    return score_fun, None, None