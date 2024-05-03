import pickle
import os
import argparse
import datetime
import jax.numpy as jnp
import time

from src.models import load_pretrained_model, compute_num_params, compute_norm_params
from src.datasets import get_test_loaders, get_output_dim
from src.datasets.cifar10 import corruption_types

from src.ood_scores.ensemble import ensemble_score_fun
from src.ood_scores.diagonal_lla import diagonal_lla_score_fun
from src.ood_scores.scod import scod_score_fun
from src.ood_scores.swag import swag_score_fun
from src.ood_scores.hm_lanczos import high_memory_lanczos_score_fun, smart_lanczos_score_fun
from src.ood_scores.lm_lanczos import low_memory_lanczos_score_fun

parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--data_path", type=str, default="../datasets/", help="root of dataset")
parser.add_argument("--ID_dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA"], default="MNIST", required=True)
parser.add_argument('--OOD_datasets', nargs='+', help='List of OOD datasets to score')
parser.add_argument("--subsample_trainset", default=None, type=int, help="Subsampling of the train datasets used to compute scores")
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--test_batch_size", default=256, type=int)
# pretrained-model hyperparams
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--model", type=str, default="LeNet", help="Pretrained model to use")
parser.add_argument("--run_name", type=str, default="good", help="Name of pretrained model to use")
parser.add_argument("--model_seed", default=420, type=int)
##############
# ood scores #
##############
parser.add_argument("--score", type=str, choices=["scod", "swag", "ensemble", "local_ensemble", "sketched_local_ensemble", "low_rank_lla", "smart_lla", "diagonal_lla"], default=None)
# lanczos
parser.add_argument("--lanczos_hm_iter", default=10, type=int, help="Lancsos high-memory iterations to run")
parser.add_argument("--lanczos_lm_iter", default=100, type=int, help="Lancsos low-mwmory iterations to run")
parser.add_argument("--n_eigenvec_hm", default=None, type=int, help="Number of eigenvectors to store. Default None set it to half of lanczos iterations")
parser.add_argument("--n_eigenvec_lm", default=None, type=int, help="Number of eigenvectors to store. Default None set it to half of lanczos iterations")
parser.add_argument("--lanczos_seed", default=0, type=int, help="Seed for the initial vector of Lanczos")
# sketch
parser.add_argument("--sketch", type=str, choices=["dense", "srft"], default=None, help="Default None means no sketch is applied (i.e. the identity matrix)")
parser.add_argument("--sketch_size", default=1000, type=int, help="Dimension of sketched vectors")
parser.add_argument("--sketch_seed", default=0, type=int, help="Seed for the sketch operator")
parser.add_argument("--sketch_padding", default=None, type=int, help="Padding for srft sketch")
parser.add_argument("--sketch_density", default=None, type=float, help="Density for dense sketch. Defalut None set it to the theoretical optimal value")
# eigenvalues vs projection
parser.add_argument("--use_eigenvals", action="store_true", required=False, default=False)
parser.add_argument("--prior_std", default=0.1, type=float, help="Scale the eigenvalues (if they are used)")
# generalized gauss newton vs hessian
parser.add_argument("--use_hessian", action="store_true", required=False, default=False)
# diagonal lla
parser.add_argument("--hutchinson_samples", default=10000, type=int, help="Only used for diagonal lla score")
parser.add_argument("--hutchinson_seed", default=1, type=int, help="Only used for diagonal lla score")
# ensemble
parser.add_argument("--ensemble_size", default=5, type=int, help="Only used for ensemble score")
#swag
parser.add_argument("--swag_n_vec", default=0, type=int, help="Only used for swag score")
parser.add_argument("--swag_diag_only", action="store_true", required=False, default=False)
parser.add_argument("--swag_lr", default=0.001, type=float)
parser.add_argument("--swag_momentum", default=0.9, type=float)
parser.add_argument("--swag_collect_interval", default=3, type=int)

# print more stuff
parser.add_argument("--verbose", action="store_true", required=False, default=False)





if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.model_seed)

    ################
    ### datasets ###
    train_loader, _, _ = get_test_loaders(
        args.ID_dataset,
        n_samples = args.subsample_trainset,
        batch_size = args.train_batch_size,
        shuffle = False,
        seed = 0,
        download = False,
        data_path = args.data_path
    )
    _, _, ID_loader = get_test_loaders(
        args.ID_dataset,
        n_samples = None,
        batch_size = args.test_batch_size,
        shuffle = False,
        seed = 0,
        download = False,
        data_path = args.data_path
    )
    print(f"Got IN-distribution dataset {args.ID_dataset} with {len(train_loader.dataset)} train data and {len(ID_loader.dataset)} test data")

    if "MNIST-R" in args.OOD_datasets:
        rotated_datasets = [f"MNIST-R{angle}" for angle in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]]
        args_dict["OOD_datasets"].remove("MNIST-R")
        args_dict["OOD_datasets"] += rotated_datasets
    if "FMNIST-R" in args.OOD_datasets:
        rotated_datasets = [f"FMNIST-R{angle}" for angle in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]]
        args_dict["OOD_datasets"].remove("FMNIST-R")
        args_dict["OOD_datasets"] += rotated_datasets
    if "CIFAR-10-C" in args.OOD_datasets:
        rotated_datasets = [f"CIFAR-10-C{severity}-{corr}" for corr in corruption_types for severity in [1,2,3,4,5] ]
        args_dict["OOD_datasets"].remove("CIFAR-10-C")
        args_dict["OOD_datasets"] += rotated_datasets
    OOD_loaders = [get_test_loaders(
        OOD_dataset,
        n_samples = None,
        batch_size = args.test_batch_size,
        shuffle = False,
        seed = 0,
        download = False,
        data_path = args.data_path,
        angle = 0,          # for rotated datasets
        corr_type = "fog",  # for corrupted datasets
        severity_level = 5, # for corrupted datasets
    )[2] for OOD_dataset in args_dict["OOD_datasets"]]
    for d, loader in zip(args_dict["OOD_datasets"], OOD_loaders):
        print(f"Got OUT-of-distribution dataset {d} with {len(loader.dataset)} test data")
    args_dict["output_dim"] = get_output_dim(args.ID_dataset)
    

    #############
    ### model ###
    model, params_dict, model_arg_dict = load_pretrained_model(
        dataset_name = args.ID_dataset,
        model_name = args.model,
        run_name = args.run_name,
        seed = args.model_seed,
        save_path = args.model_save_path
    )
    args_dict["likelihood"] = model_arg_dict["likelihood"]
    print(f"Loaded {args.model} with {compute_num_params(params_dict['params'])} parameters of norm {compute_norm_params(params_dict['params']):.2f}")

    ###################
    ### define score ##

    if args.score in ["local_ensemble", "sketched_local_ensemble", "low_rank_lla", "smart_lla"]:
        # with or withouth eigenvalues
        if args.score in ["local_ensemble", "sketched_local_ensemble"]:
            args_dict['use_eigenvals'] = False
        elif args.score in ["low_rank_lla", "smart_lla"]:
            args_dict['use_eigenvals'] = True
        # with or without sketching
        if args.score in ["sketched_local_ensemble", "smart_lla"] and args_dict["sketch"] is None:
            args_dict["sketch"] = "srft"
        # high memory only methods should not perform low memory iterations, and same for reverse
        if args.score in ["local_ensemble", "low_rank_lla"]:
            args_dict['lanczos_lm_iter'] = 0
        elif args.score in ["sketched_local_ensemble"]:
            args_dict['lanczos_hm_iter'] = 0

    # number of "good" vectors out of Lanczsos is 90% by default
    if args_dict["n_eigenvec_hm"] is None:
        args_dict["n_eigenvec_hm"] = int(0.9 * args_dict["lanczos_hm_iter"])
    if args_dict["n_eigenvec_lm"] is None:
        args_dict["n_eigenvec_lm"] = int(0.9 * args_dict["lanczos_lm_iter"])

    # set reasonable srft sketching padding to reduce prime factorization max value (needed by jax fft implementation)
    if args.sketch == "srft" and args.sketch_padding is None:
        args_dict["sketch_padding"] = 0
        if args.model == "LeNet":
            if args.ID_dataset in ["MNIST", "FMNIST"]:
                args_dict["sketch_padding"] = 10 # params 44426 -> 44436 = 2^2 × 3 × 7 × 23^2
        elif args.model == "GoogleNet":
            if args.ID_dataset in ["CIFAR-10", "SVHN"]:
                args_dict["sketch_padding"] = 8 # params 259338 -> 259346 = 2 × 31 × 47 × 89
        elif args.model == "ResNet":
            if args.ID_dataset in ["CIFAR-10", "SVHN"]:
                args_dict["sketch_padding"] = 6 # params 272378 -> 272384 = 2^11 × 7 × 19
            elif args.ID_dataset == "CIFAR-100":
                args_dict["sketch_padding"] = 12 # params 278228 -> 278240 = 2^5 × 5 × 37 × 47
        elif args.model == "ResNet50":
            if args.ID_dataset in ["CelebA"]:
                args_dict["sketch_padding"] = 11 # params 5327857 -> 5327868 = 2^2 × 3 × 7^2 × 13 × 17 × 41

    if args.score == "ensemble":
        params_dicts_list = [params_dict]
        for i in range(args.model_seed + 1, args.model_seed + args.ensemble_size):
            _, params_dict , _= load_pretrained_model(
                dataset_name = args.ID_dataset,
                model_name = args.model,
                run_name = args.run_name,
                seed = i,
                save_path = args.model_save_path
            )
            params_dicts_list.append(params_dict)
        score_fun = ensemble_score_fun(model, params_dicts_list)
        eigenval = []
        approx_quadratic_form, quadratic_form = None, None
    elif args.score == "diagonal_lla":
        score_fun, quadratic_form, approx_quadratic_form = diagonal_lla_score_fun(model, params_dict, train_loader, args_dict)
        eigenval = []
    elif args.score == "scod":
        args_dict['use_eigenvals'] = True
        score_fun, eigenval, approx_quadratic_form = scod_score_fun(model, params_dict, train_loader, args_dict, use_eigenvals=True)
        quadratic_form = None
    elif args.score == "swag":
        score_fun, _, _ = swag_score_fun(
            model, params_dict, train_loader, args_dict,
            #diag_only=False, max_num_models=10, swa_c_epochs=1, swa_c_batches=None,
            diag_only = args_dict['swag_diag_only'], 
            max_num_models = args_dict['swag_n_vec'], 
            swa_c_epochs = None, swa_c_batches = args_dict['swag_collect_interval'],
            swa_lr = args_dict['swag_lr'], 
            momentum = args_dict['swag_momentum'], 
            wd=0.0 #1e-6
        )
        eigenval = []
        approx_quadratic_form, quadratic_form = None, None
    else:
        if args_dict['lanczos_hm_iter']==0:
            # low memory lanczos methods
            score_fun, eigenval, approx_quadratic_form, quadratic_form = low_memory_lanczos_score_fun(
                model, 
                params_dict, 
                train_loader, 
                args_dict, 
                use_eigenvals = args_dict['use_eigenvals']
            )
        else:
            # high memory lanczos methods
            if args_dict['lanczos_lm_iter']==0:
                # standard high memory lanczos
                score_fun, eigenval, approx_quadratic_form, quadratic_form = high_memory_lanczos_score_fun(
                    model, 
                    params_dict, 
                    train_loader, 
                    args_dict, 
                    use_eigenvals = args_dict['use_eigenvals']
                )
            else:
                # high memory lanczos is used as preconditioner to smart low memory lanczos
                score_fun, eigenval, approx_quadratic_form, quadratic_form = smart_lanczos_score_fun(
                    model, 
                    params_dict, 
                    train_loader, 
                    args_dict, 
                    use_eigenvals = args_dict['use_eigenvals']
                )
    if args.verbose:
        print(f"Eigenvalues: {eigenval}")

    ######################
    ### compute scores ###
    approx_quadratic_form = None # skip computation of approx quadratic form
    compute_true_quadratic_form = False # skip computation of true quadratic form
    scores_dict = {
        "eigenvals": jnp.array(eigenval),
        "args_dict": args_dict
    }
    for distribution, loader in [("ID", ID_loader), *zip(args_dict["OOD_datasets"], OOD_loaders)]:
        start = time.time()
        done = 0
        scores_dict[distribution] = []
        if approx_quadratic_form is not None:
            scores_dict[f"{distribution}_QF"] = []
            scores_dict[f"{distribution}_QFapprox"] = []

        for batch in loader:
            #if done > 200:
            #    break
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())
            start_batch = time.time()
            # here you apply score_fun to a batch of datapoints
            batch_scores = score_fun(X)
            scores_dict[distribution].append(batch_scores)
            if approx_quadratic_form is not None:
                fake = approx_quadratic_form(X)
                scores_dict[f"{distribution}_QFapprox"].append(fake)
                if compute_true_quadratic_form and done<1:
                    # real is very expensive to compute, and does not depend on the score
                    for i in range(4):
                        small_X = X[i*4 : (i+1)*4]
                        real = quadratic_form(small_X)
                        scores_dict[f"{distribution}_QF"].append(real)
            #print(f"{distribution} - scores {batch_scores}, computed in {time.time()-start:.3f}s")
            done += X.shape[0]
            if args.verbose:
                print(f"{done}/{len(loader.dataset)} in {time.time()-start_batch:.3f}s")
        print(f"Computed {distribution} scores in {time.time()-start:.3f} seconds")

        scores_dict[distribution] = jnp.concatenate(scores_dict[distribution], axis=0)
        if approx_quadratic_form is not None:
            scores_dict[f"{distribution}_QFapprox"] = jnp.concatenate(scores_dict[f"{distribution}_QFapprox"], axis=0)
            if compute_true_quadratic_form:
                scores_dict[f"{distribution}_QF"] = jnp.concatenate(scores_dict[f"{distribution}_QF"], axis=0)

    ###################
    ### save scores ###
    experiment_name = f"scores_"
    if args.subsample_trainset is not None:
        experiment_name += f"subsample{args.subsample_trainset}_"

    if args.score == "ensemble":
        experiment_name += f"ensemble_size{args.ensemble_size}"
    elif args.score == "diagonal_lla":
        experiment_name += f"diagonal_lla_sample{args.hutchinson_samples}"
    elif args.score == "scod":
        experiment_name += f"scod_HMsize{args.n_eigenvec_hm}"
    elif args.score == "swag":
        experiment_name += f"swag_vec{args.swag_n_vec}_mom{args.swag_momentum}_collect{args.swag_collect_interval}"
        if args.swag_diag_only:
            experiment_name += "_diag"
    else:
        if args.use_eigenvals:
            experiment_name += "eig_"
        if args.use_hessian:
            experiment_name += "hess_"
        #lanczos params
        experiment_name += f"lanczos_seed{args.lanczos_seed}_size_HM{args.n_eigenvec_hm}of{args.lanczos_hm_iter}_LM{args.n_eigenvec_lm}of{args.lanczos_lm_iter}"
        #sketch params
        if args.sketch is not None:
            experiment_name += f"_sketch_{args.sketch}_seed{args.sketch_seed}_size{args.sketch_size}"
    print(f"Saving with name -> {experiment_name}\n\n")
    pickle.dump(scores_dict, open(f"{args.model_save_path}/{args.ID_dataset}/{args.model}/seed_{args.model_seed}/{args.run_name}_{experiment_name}.pickle", "wb"))