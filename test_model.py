import pickle
import os
import argparse
import datetime
import jax
import jax.numpy as jnp
import time
import tqdm
from functools import partial

from src.datasets import dataloader_from_string
from src.models import pretrained_model_from_string, compute_num_params, compute_norm_params

from src.training.losses import log_gaussian_log_loss, cross_entropy_loss, multiclass_binary_cross_entropy_loss
from src.training.losses import get_loss_function



parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint used for training. None means all")
parser.add_argument("--uci_type", type=str, choices=["concrete", "boston", "energy", "kin8nm", "wine", "yacht"], default=None)
parser.add_argument("--batch_size", type=int, default=512)

# model hyperparams
parser.add_argument("--model", type=str, default="LeNet", help="Model architecture.")

# training hyperparams
parser.add_argument("--seed", default=420, type=int)

# storage
parser.add_argument("--run_name", default="good")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where the pretrained model is saved")

parser.add_argument("--verbose", action="store_true", required=False, default=False)



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


    ###############
    ### dataset ###
    train_loader, valid_loader, test_loader = dataloader_from_string(
        args.dataset,
        n_samples = args.n_samples,
        batch_size = args.batch_size,
        shuffle = False,
        seed = args.seed,
        download = False,
        data_path = args.data_path
    )
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}, Test set size {len(test_loader.dataset)}")


    #############
    ### model ###
    model, params_dict, args_dict = pretrained_model_from_string(
        model_name = args.model,
        dataset_name = args.dataset,
        n_samples = args.n_samples,
        run_name = args.run_name,
        seed = args.seed,
        save_path = args.model_save_path
    )
    print(f"Loaded {args.model} with {compute_num_params(params_dict['params'])} parameters of norm {compute_norm_params(params_dict['params']):.2f}")

    
    ###############
    ### testing ###  
    def get_loss(
            model, 
            likelihood,
            class_frequencies=None
        ):
        if likelihood == "regression":
            negative_log_likelihood = log_gaussian_log_loss
            extra_stats_function = lambda preds, y : (preds-y)**2                                # sum of squared error
        elif likelihood == "classification":
            negative_log_likelihood = cross_entropy_loss
            extra_stats_function = lambda preds, y : preds.argmax(axis=-1) == y.argmax(axis=-1)    # accuracy
        elif likelihood == "binary_multiclassification":
            negative_log_likelihood = lambda preds, y: multiclass_binary_cross_entropy_loss(preds, y, class_frequencies=class_frequencies)
            extra_stats_function = lambda preds, y : (preds>0.) == (y==1)                 # multiclass accuracy
        else:
            raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
        
        if model.has_batch_stats:
            @jax.jit
            def calculate_loss(params_dict, X, Y): 
                preds = model.apply_test(params_dict['params'], params_dict['batch_stats'], X)
                #print("\n\n",preds, "\n", Y)
                loss = negative_log_likelihood(preds, Y)
                acc_or_sse = extra_stats_function(preds, Y)
                return loss, acc_or_sse
        else:
            @jax.jit
            def calculate_loss(params_dict, X, Y):
                preds = model.apply_test(params_dict['params'], X)
                loss = negative_log_likelihood(preds, Y)
                acc_or_sse = extra_stats_function(preds, Y)
                return loss, acc_or_sse
        return calculate_loss
    #_, loss_function_test = get_loss_function(
    #    model,
    #    likelihood = args_dict["likelihood"],
    #    class_frequencies = train_loader.dataset.dataset.dataset.class_frequencies if args_dict["likelihood"]=="binary_multiclassification" else None
    #)
    calculate_loss = get_loss(
        model, 
        args_dict["likelihood"], 
        #class_frequencies = train_loader.dataset.dataset.dataset.class_frequencies if args_dict["likelihood"]=="binary_multiclassification" else None
        class_frequencies = train_loader.dataset.dataset.class_frequencies if args_dict["likelihood"]=="binary_multiclassification" else None
        )

    def get_stats(params_dict, loader):
        loss = 0.
        acc_or_sse = []
        start_time = time.time()
        loader_bar = tqdm.tqdm(loader) if args_dict["verbose"] else loader
        for batch in loader_bar:
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            batch_loss, batch_acc_or_sse = calculate_loss(params_dict, X,Y)
            #batch_loss, batch_acc_or_sse = loss_function_test(params_dict["params"], params_dict["batch_stats"], X,Y)
            
            loss += batch_loss.item()
            acc_or_sse.append(batch_acc_or_sse)
            
        acc_or_sse = jnp.concatenate(acc_or_sse)
        loss /= len(loader)
        return loss, acc_or_sse, time.time() - start_time
    
    predictions = {}
    #for loader_type, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
    for loader_type, loader in [("valid", valid_loader), ("test", test_loader), ("train", train_loader)]:
        if len(loader)==0:
            print(f"empty {loader_type} loader")
            continue
        loss, acc_or_sse, duration = get_stats(params_dict, loader)
        #print(acc_or_sse.shape, acc_or_sse[:30])
        predictions[loader_type] = acc_or_sse
        predictions[f"{loader_type} loss"] = loss
        acc_or_sse = acc_or_sse.mean(axis=0)
        #print(acc_or_sse.shape)
        acc_or_sse = f"{acc_or_sse:.3f}" if args_dict["likelihood"] != "binary_multiclassification" else acc_or_sse #[f"{a:.3f}" for a in acc_or_sse]
        if args_dict["likelihood"] in ["classification", "binary_multiclassification"]:
            acc_label = "accuracy"
        elif args_dict["likelihood"] == "regression":
            acc_label = "mse"
        print(f"{loader_type} stats\t - loss={loss:.3f}, {acc_label}={acc_or_sse}, time={duration:.3f}s")

    ####################################
    ### save params and dictionaries ###
    # first folder is dataset
    save_folder = f"{args.model_save_path}/{args.dataset}"
    if args.n_samples is not None:
        save_folder += f"_samples{args.n_samples}"
    # second folder is model
    save_folder += f"/{args.model}"
    # third folder is seed
    save_folder += f"/seed_{args.seed}"
    if args.run_name is not None:
        save_name = f"{args.run_name}"
    else:
        save_name = f"started_{now_string}"
    pickle.dump(predictions, open(f"{save_folder}/{save_name}_predictions.pickle", "wb"))