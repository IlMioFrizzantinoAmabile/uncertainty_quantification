import pickle
import os
import argparse
import datetime
import jax
import jax.numpy as jnp
import time
import tqdm
from functools import partial

from src.datasets import get_test_loaders, get_output_dim
from src.models import load_pretrained_model, compute_num_params, compute_norm_params, has_batchstats

from src.training.loss import calculate_loss_without_batchstats, calculate_loss_with_batchstats
from src.training.loss import log_gaussian_log_loss, cross_entropy_loss, multiclass_binary_cross_entropy_loss



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



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


    ###############
    ### dataset ###
    train_loader, valid_loader, test_loader = get_test_loaders(
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
    model, params_dict, args_dict = load_pretrained_model(
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
    likelihood=args_dict["likelihood"]
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = multiclass_binary_cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
    
    #if params_dict['batch_stats']:
    if has_batchstats(model):
        print("with")

        def calculate_loss(model, params_dict, X, Y): 
            preds = model.apply(
                {'params': params_dict['params'], 'batch_stats': params_dict['batch_stats']},
                X,
                train=False,
                mutable=False)
            loss = negative_log_likelihood(preds, Y)
            if likelihood == "regression":
                sse = (preds-Y)**2
                return loss, sse
            elif likelihood == "classification":
                acc = preds.argmax(axis=-1) == Y.argmax(axis=-1)
                return loss, acc
            elif likelihood == "binary_multiclassification":
                num_classes = preds.shape[1]
                #acc = jnp.sum((preds>0.) == (y==1)) / num_classes
                correct = (preds>0.) == (Y==1)
                #print(correct.shape)
                acc = jnp.sum(correct, axis=1)
                #print(acc.shape)
                return loss, acc
    else:
        print("without")

        def calculate_loss(model, params_dict, X, Y):
            preds = model.apply(params_dict['params'], X)
            loss = negative_log_likelihood(preds, Y)
            if likelihood == "regression":
                sse = (preds-Y)**2
                return loss, sse
            elif likelihood == "classification":
                acc = preds.argmax(axis=-1) == Y.argmax(axis=-1)
                return loss, acc
            elif likelihood == "binary_multiclassification":
                num_classes = preds.shape[1]
                #acc = jnp.sum((preds>0.) == (y==1)) / num_classes
                acc = (preds>0.) == (Y==1)
                return loss, acc
    #calculate_loss_jit = jax.jit(calculate_loss)

    def get_stats(model, params_dict, loader):
        loss = 0.
        #acc_or_sse = 0. if args_dict["likelihood"] != "binary_multiclassification" else jnp.zeros((get_output_dim(args.dataset), ))
        acc_or_sse = []
        start_time = time.time()
        for batch in tqdm.tqdm(loader):
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            #batch_loss, batch_acc_or_sse = calculate_loss_jit(model, params_dict, X,Y)
            batch_loss, batch_acc_or_sse = calculate_loss(model, params_dict, X,Y)
            
            loss += batch_loss.item()
            #acc_or_sse += batch_acc_or_sse/X.shape[0]
            acc_or_sse.append(batch_acc_or_sse)
            #print(len(batch_acc_or_sse), batch_acc_or_sse[0])
        #acc_or_sse /= len(loader)
        print("aaa", len(acc_or_sse), acc_or_sse.shape)
        acc_or_sse = jnp.concatenate(acc_or_sse)
        loss /= len(loader)
        return loss, acc_or_sse, time.time() - start_time
    
    predictions = {}
    for loader_type, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
        loss, acc_or_sse, duration = get_stats(model, params_dict, loader)
        print(acc_or_sse.shape, acc_or_sse[:30])
        predictions[loader_type] = acc_or_sse
        predictions[f"{loader_type} loss"] = loss
        acc_or_sse = acc_or_sse.mean(axis=0)
        print(acc_or_sse.shape)
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