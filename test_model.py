import os
import argparse
import datetime
import jax
import jax.numpy as jnp
import time
import tqdm

from src.datasets import get_test_loaders
from src.models import load_pretrained_model, compute_num_params, compute_norm_params

from src.training.loss import calculate_loss_without_batchstats, calculate_loss_with_batchstats



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
    
    if params_dict['batch_stats']:
        calculate_loss = lambda X, Y : calculate_loss_with_batchstats(
            model, 
            params_dict['params'], 
            params_dict['batch_stats'], 
            X, 
            Y, 
            train=False, 
            likelihood=args_dict["likelihood"]
        )
    else:
        calculate_loss = lambda X, Y :  calculate_loss_without_batchstats(
            model, 
            params_dict['params'], 
            X, 
            Y, 
            likelihood=args_dict["likelihood"]
        )
    calculate_loss_jit = jax.jit(calculate_loss)

    def get_stats(loader):
        loss, acc_or_sse = 0., 0.
        start_time = time.time()
        for batch in tqdm.tqdm(loader):
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            batch_loss, (batch_acc_or_sse, _) = calculate_loss_jit(X,Y)
            
            loss += batch_loss.item()
            acc_or_sse += batch_acc_or_sse/X.shape[0]
        acc_or_sse /= len(loader)
        loss /= len(loader)
        return loss, acc_or_sse, time.time() - start_time
    
    for loader_type, loader in [("train", train_loader), ("valid", valid_loader), ("test ", test_loader)]:
        loss, acc_or_sse, duration = get_stats(loader)
        if args_dict["likelihood"] in ["classification", "binary_multiclassification"]:
            print(f"{loader_type} stats\t - loss={loss:.3f}, accuracy={acc_or_sse:.3f}, time={duration:.3f}s")
        elif args_dict["likelihood"] == "regression":
            print(f"{loader_type} stats\t - loss={loss:.3f}, mse={acc_or_sse:.3f}, time={duration:.3f}s")