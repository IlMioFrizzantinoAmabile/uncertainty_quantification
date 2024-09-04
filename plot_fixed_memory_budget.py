import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from jax import flatten_util
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

from src.datasets.cifar10 import corruption_types
corruption_types = [
    "brightness",
    #"contrast",
    "defocus_blur",
    "elastic_transform",
    #"fog",
    "frost",
    "gaussian_blur",
    #"gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    #"shot_noise",
    "snow",
    "spatter",
    #"speckle_noise",
    "zoom_blur",
]



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "ImageNet"], default="MNIST")
parser.add_argument("--model", type=str, default=None, help="Model architecture.")
parser.add_argument("--budget", default=3, type=int)

# storage
parser.add_argument("--run_name", default="good")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where the pretrained model is saved")



if __name__ == "__main__":
    args = parser.parse_args()


    dataset = args.dataset
    budget = args.budget

    run_name = args.run_name

    if args.model is None:
        if dataset=="MNIST":
            model = "MLP_depth1_hidden20"
        elif dataset=="FMNIST":
            model = "LeNet"
        if dataset=="CIFAR-10":
            model = "ResNet"
        if dataset=="CelebA":
            model = "VAN_tiny"
            #model = "VAN_large"
        if dataset=="ImageNet":
            model = "VAN_large"
    else:
        model = args.model


    save_folder = f"{args.model_save_path}/{dataset}"
    # second folder is model
    if model == "MLP":
        save_folder += f"/MLP_depth{1}_hidden{20}"
    else:
        save_folder += f"/{model}"
    #save_folder += f"/seed_{seed}"

    if dataset == "MNIST":
        ood_datasets = ["FMNIST", "KMNIST", "MNIST-R"]
    elif dataset == "FMNIST":
        ood_datasets = ["MNIST", "FMNIST-R"]
    elif dataset == "CIFAR-10":
        ood_datasets = ["SVHN", "CIFAR-100", "CIFAR-10-C"]
    elif dataset == "SVHN":
        ood_datasets = ["CIFAR-10"]
    elif dataset == "CelebA":
        ood_datasets = ["FOOD101", "CelebA-Mustache", "CelebA-Bald", "CelebA-Eyeglasses"]
        ood_datasets_xlabel = ["FOOD-101", "Mustache only", "Bald only", "Eyeglasses only"]
        #ood_datasets = ["CelebA-Mustache", "CelebA-Bald", "CelebA-Eyeglasses"]
        #ood_datasets_xlabel = ["Mustache only", "Bald only", "Eyeglasses only"]
    elif dataset == "ImageNet":
        ood_datasets = ["SVHN-256", "FOOD101-256", "ImageNet-classout"]
        ood_datasets_xlabel = ["SVHN", "FOOD-101"]


    if "MNIST-R" in ood_datasets:
        rotated_datasets = [f"MNIST-R{angle}" for angle in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]]
        ood_datasets.remove("MNIST-R")
        ood_datasets += rotated_datasets
    if "FMNIST-R" in ood_datasets:
        rotated_datasets = [f"FMNIST-R{angle}" for angle in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]]
        ood_datasets.remove("FMNIST-R")
        ood_datasets += rotated_datasets
        ood_datasets_xlabel = [f"Rotated {angle}°" for angle in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]]
        ood_datasets_xlabel = [ood_datasets[0]] + ood_datasets_xlabel
    if "CIFAR-10-C" in ood_datasets:
        rotated_datasets = [f"CIFAR-10-C{severity}-{corr}" for corr in corruption_types for severity in [5]] #[1,2,3,4,5] ]
        ood_datasets.remove("CIFAR-10-C")
        ood_datasets += rotated_datasets
        ood_datasets_xlabel = ["SVHN", "CIFAR-100", "elastic transform", "defocus blur", "gaussian blur", "glass blur", "impulse noise", "motion blur", "pixelate", "saturate", "snow", "spatter", "zoom blur"]
        ood_datasets_xlabel = ["SVHN", "CIFAR-100", "Brightness", "Defocus blur", "Elastic transf", "Frost", "Gaussian blur", "Glass blur", "Impulse noise", "Jpeg compress", "Motion blur",
                "Pixelate", "Saturate", "Snow", "Spatter", "Zoom blur"]
    if "ImageNet-classout" in ood_datasets:
        classout_datasets = [f"ImageNet-{c}" for c in ["pineapple", "carbonara", "menu", "volcano", "flamingo", "triceratops", "odometer", "lighter", "castle", "parachute"]]
        ood_datasets.remove("ImageNet-classout")
        ood_datasets += classout_datasets
        ood_datasets_xlabel += ["pineapple", "carbonara", "menu", "volcano", "flamingo", "triceratops", "odometer", "lighter", "castle", "parachute"]

        
    def auroc(scores_id, scores_ood):
        labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
        labels[len(scores_id) :] = 1
        scores = np.concatenate([scores_id, scores_ood])
        return roc_auc_score(labels, scores)

    def plot_roc(scores_id, scores_ood, label=None, c=None, alpha=1.):
        labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
        labels[len(scores_id) :] = 1
        scores = np.concatenate([scores_id, scores_ood])
        fpr, tpr, thresholds = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=label, c=c, alpha=alpha)

    def get_experiment_name(
            subsample_trainset = None, 
            score = None,
            ensemble_size = 5,
            hutchinson_samples = 1000,
            lanczos_seed = 0,
            lanczos_hm_iter = 10,
            lanczos_lm_iter = 10,
            n_eigenvec_hm = 5,
            n_eigenvec_lm = 5,
            use_eigenvals = False,
            use_hessian = False,
            sketch = None,
            sketch_seed = 0,
            sketch_size = 1000
        ):

        experiment_name = f"scores_"

        if score == "ensemble":
            experiment_name += f"ensemble_size{ensemble_size}"
            return experiment_name
        elif score == "swag":
            experiment_name += f"swag_vec{lanczos_hm_iter}_mom{0.99}_collect{1000}"
            return experiment_name
        
        if subsample_trainset is not None:
            experiment_name += f"subsample{subsample_trainset}_"
            
        if score == "diagonal_lla":
            experiment_name += f"diagonal_lla_sample{hutchinson_samples}"
            return experiment_name
        elif score == "scod":
            experiment_name += f"scod_HMsize{n_eigenvec_hm}"
        else:
            if use_eigenvals:
                experiment_name += "eig_"
            if use_hessian:
                experiment_name += "hess_"
            #lanczos params
            experiment_name += f"lanczos_seed{lanczos_seed}_size_HM{n_eigenvec_hm}of{lanczos_hm_iter}_LM{n_eigenvec_lm}of{lanczos_lm_iter}"
            #sketch params
            if sketch is not None:
                experiment_name += f"_sketch_{sketch}_seed{sketch_seed}_size{sketch_size}"
                
        return experiment_name

    params_file_path = f"{save_folder}/seed_1/{run_name}_params.pickle"
    params_dict = pickle.load(open(params_file_path, 'rb'))
    params = params_dict['params']
    P = flatten_util.ravel_pytree(params)[0].shape[0]
    print(f"model has {P} parameters")


    stats_file_path = f"{save_folder}/seed_1/{run_name}_stats.pickle"
    stats_dict = pickle.load(open(stats_file_path, 'rb'))
    if len(stats_dict['train_acc_or_mse']):
        if dataset != "CelebA":
            print(f"Accuracies for seed 1:\t train= {stats_dict['train_acc_or_mse'][-1]:.3f}, test= {stats_dict['valid_acc_or_mse'][-1]:.3f}")
        else:
            print(f"Accuracies for seed 1:\n\t train= {stats_dict['train_acc_or_mse'][-1]}, \n\t test= {stats_dict['valid_acc_or_mse'][-1]}")



    if run_name == "epoch0":
        model_seeds = [1, 2, 3]
        lanczos_seeds = [1]
    elif run_name == "good":
        model_seeds = [1, 2, 3]
        lanczos_seeds = [1]# 1, 2]
    elif run_name == "e5lr3":
        model_seeds = [2]
        lanczos_seeds = [1]# 1, 2]
    if dataset=="FMNIST":
        subsample_trainsets = [60000, 10000, 1000, 100]
        subsample_trainsets = [60000]
        lanczos_hm_iters = [10, 100, 1000]#, 10000]
        #subsample_trainsets = [100]  # FMNIST LeNet
    if dataset=="MNIST":
        subsample_trainsets = [50000, 10000, 1000, 100]
        subsample_trainsets = [50000]
        lanczos_hm_iters = [10, 100, 1000, 10000]
        #subsample_trainsets = [100]  # MNIST MLP
    if dataset=="CIFAR-10":
        subsample_trainsets = [10000, 1000, 100, 10]
        subsample_trainsets = [10000]
        lanczos_hm_iters = [10, 100, 1000]
    if dataset=="SVHN":
        subsample_trainsets = [1000, 100, 10]
        lanczos_hm_iters = [10, 100, 1000]
    if dataset=="CelebA":
        subsample_trainsets = [10000]#100, 1000]
        lanczos_hm_iters = [100]
    if dataset=="ImageNet":
        subsample_trainsets = [100000]


    alpha = 0.07
    sketch_size = 10000
    smart = False
    use_all_eigenvectors = False
    #sns.set_style('whitegrid')
    markers = ["P", "v", "^", "X", "o", "*", '.','.', ',']
    names = [
        #"SLU", 
        #"LLA / LE / LE-h",
        #"LLA",
        #"LE",
        #"LE-h", 
        #"LLA-d", 
        #"SCOD", 
        "SWAG",
        #"DE"
    ]
    if dataset=="MNIST": 
        if budget == 3: #seeds 1-5
            experiment_names = [ #mem budget 3
                "lanczos_seed1_size_HM0of0_LM40of45_sketch_srft_seed0_size1000",
                #"lanczos_seed1_size_HM1of2_LM27of30_sketch_srft_seed0_size1000",
                "eig_lanczos_seed1_size_HM2of3_LM0of0", 
                #"lanczos_seed1_size_HM2of3_LM0of0", 
                #"hess_lanczos_seed1_size_HM2of3_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize3",
                "swag_vec3_mom0.99_collect1000",
                "ensemble_size3"]
        if budget == 10: #seeds 1-5
            experiment_names = [ #mem budget 10
                "lanczos_seed1_size_HM0of0_LM135of150_sketch_srft_seed0_size1000",
                #"lanczos_seed1_size_HM3of4_LM94of105_sketch_srft_seed0_size1000",
                "eig_lanczos_seed1_size_HM9of10_LM0of0", 
                #"lanczos_seed1_size_HM9of10_LM0of0", 
                #"hess_lanczos_seed1_size_HM9of10_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize10",
                "swag_vec10_mom0.99_collect1000",
                "ensemble_size10"]
    elif dataset=="FMNIST": 
        if budget == 3: #seeds 1-10
            experiment_names = [ #mem budget 3
                "lanczos_seed1_size_HM0of0_LM118of132_sketch_srft_seed0_size1000",
                #"lanczos_seed1_size_HM1of2_LM79of88_sketch_srft_seed0_size1000",
                "eig_lanczos_seed1_size_HM2of3_LM0of0", 
                #"lanczos_seed1_size_HM2of3_LM0of0", 
                #"hess_lanczos_seed1_size_HM2of3_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize3",
                "swag_vec3_mom0.99_collect1000",
                "ensemble_size3"]
        if budget == 10: #seeds 1-10
            experiment_names = [ #mem budget 10
                "lanczos_seed1_size_HM0of0_LM396of440_sketch_srft_seed0_size1000",
                #"lanczos_seed1_size_HM3of4_LM277of308_sketch_srft_seed0_size1000",
                #"lanczos_seed1_size_HM7of8_LM118of132_sketch_srft_seed0_size1000",
                "eig_lanczos_seed1_size_HM9of10_LM0of0", 
                #"lanczos_seed1_size_HM9of10_LM0of0", 
                #"hess_lanczos_seed1_size_HM9of10_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize10",
                "swag_vec10_mom0.99_collect1000",
                "ensemble_size10"]
    elif dataset=="CIFAR-10":
        if budget == 3: #seeds 1-5
            experiment_names = [
                "lanczos_seed1_size_HM0of0_LM72of81_sketch_srft_seed0_size10000", 
                #"lanczos_seed1_size_HM4of5_LM9of10_sketch_srft_seed0_size10000", 
                "eig_lanczos_seed1_size_HM2of3_LM0of0", 
                #"lanczos_seed1_size_HM2of3_LM0of0", 
                #"hess_lanczos_seed1_size_HM2of3_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize3", # wrong  3 running
                "swag_vec3_mom0.99_collect1000", 
                "ensemble_size3"]
        elif budget == 10: #seeds 1-5
            experiment_names = [
                "lanczos_seed1_size_HM0of0_LM45of50_sketch_srft_seed0_size50000", 
                #"lanczos_seed1_size_HM3of4_LM31of35_sketch_srft_seed0_size50000",  
                "eig_lanczos_seed1_size_HM9of10_LM0of0", 
                #"lanczos_seed1_size_HM9of10_LM0of0", 
                #"hess_lanczos_seed1_size_HM9of10_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize10", # wrong  3 running
                "swag_vec10_mom0.99_collect1000", 
                "ensemble_size8"]                            
    elif dataset=="CelebA": 
        if budget == 3: # seeds buoni 1 3 5 (+ 6 7 10)
            experiment_names = [ #mem budget 3
                "lanczos_seed1_size_HM0of0_LM90of100_sketch_srft_seed0_size10000",
                "eig_lanczos_seed1_size_HM2of3_LM0of0", 
                #"lanczos_seed1_size_HM2of3_LM0of0", 
                #"hess_lanczos_seed1_size_HM2of3_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize3",  
                "swag_vec3_mom0.99_collect100",
                "ensemble_size3"]
            
            experiment_names = [ #mem budget 3
                "swag_vec3_mom0.99_collect100",
                ]
        if budget == 10:
            experiment_names = [ #mem budget 10
                "lanczos_seed1_size_HM0of0_LM90of100_sketch_srft_seed0_size10000",
                "eig_lanczos_seed1_size_HM9of10_LM0of0", 
                #"lanczos_seed1_size_HM9of10_LM0of0", 
                #"hess_lanczos_seed1_size_HM9of10_LM0of0",
                "diagonal_lla_sample10000", 
                "scod_HMsize10",  
                "swag_vec10_mom0.99_collect100",
                "ensemble_size10"]
    elif dataset=="ImageNet":
        if budget == 3:
            experiment_names = [
                "lanczos_seed1_size_HM0of0_LM9of10_sketch_srft_seed0_size10000000",
                "lanczos_seed1_size_HM2of3_LM0of0"
            ]
    all_aurocs = {name : [] for name in names}

    for subsample_trainset in subsample_trainsets:
        
        c = 0
        if dataset=="CIFAR-10":
            fig = plt.figure(figsize=(9,7))
        else:
            fig = plt.figure(figsize=(5,4))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
        print(f"data size {subsample_trainset}")

        for name, experiment_name in zip(names, experiment_names):

            if "swag" in experiment_name or "ensemble" in experiment_name:
                experiment_name = f"scores_{experiment_name}"
            else:
                experiment_name = f"scores_subsample{subsample_trainset}_{experiment_name}"

            aurocs, eigss = [], []

            if "ensemble" in experiment_name:
                model_seeds = [1, 4, 7]  if budget==3 else [1]
            else:
                if dataset == "MNIST":
                    model_seeds = list(range(1,6)) 
                if dataset == "FMNIST":
                    model_seeds = list(range(1,11)) 
                if dataset == "CIFAR-10":
                    model_seeds = [1,2,3,4,5] 
                if dataset == "CelebA":
                    model_seeds = [1, 2, 3]
                    model_seeds = [1]
                if dataset == "ImageNet":
                    model_seeds = [1]
        
            #model_seeds = [1,2,3]
            for model_seed in model_seeds:
                    
                
                scores_file_path = f"{save_folder}/seed_{model_seed}/{run_name}_{experiment_name}.pickle"
                scores_dict = pickle.load(open(scores_file_path, 'rb'))

                aaa = []
                for ood_dataset in ood_datasets:
                    try:
                        aaa.append(auroc(scores_dict['ID'], scores_dict[ood_dataset]))
                        #plt.figure()
                        #sns.kdeplot(np.array(scores_dict["ID"]), bw_method=0.5, color="C0", label="ID")
                        #sns.kdeplot(np.array(scores_dict[ood_dataset]), bw_method=0.5, color="C1", label=ood_dataset)
                        #plt.legend()
                        #plt.savefig(f"figures/{dataset}_budget{budget}_{name}.pdf", bbox_inches='tight')
                    except ValueError:
                        aaa.append(0.)
                aurocs.append(np.array(aaa))
                #try:
                #    aurocs.append(np.array([auroc(scores_dict['ID'], scores_dict[ood_dataset]) for ood_dataset in ood_datasets]))
                #except ValueError:
                #    aurocs.append(np.array([0. for ood_dataset in ood_datasets]))

            aurocs = np.stack(aurocs, axis=0)
            #print(f"Auroc means = {aurocs.mean(axis=0)}, stds = {aurocs.std(axis=0)}")
            #print(f"Auroc means = {aurocs.mean(axis=0)}")
            color = f"C{c}"
            #label = lanczos_hm_iter if lanczos_hm_iter!=0 else lanczos_lm_iter
            #label = f"{score} vec {label if use_all_eigenvectors else int(0.9*label)}"
            #label = score_name
            plt.plot(
                aurocs.mean(axis=0),
                label = name,
                marker= markers[c], #'o',
                color=color,
                alpha = 0.8,
                linestyle="dotted", 
                markersize=7
                )
            plt.fill_between(list(range(len(aurocs.mean(axis=0)))), aurocs.mean(axis=0) + aurocs.std(axis=0), aurocs.mean(axis=0) - aurocs.std(axis=0), alpha=alpha, color=color)
            all_aurocs[name].append(aurocs.mean(axis=0))
            all_aurocs[name].append(aurocs.std(axis=0))
            

            #if dataset=="CelebA":
            #    print(f"{aurocs.mean():.2f} ({aurocs.std(axis=0).mean():.2f})- {name}   - {experiment_name}")
            #print(aurocs.shape)
            if dataset=="CIFAR-10" or dataset=="MNIST":
                print([f"{aa.mean():.2f} ({aa.std(axis=0).mean():.2f})" for aa in (aurocs[:, :1], aurocs[:, 1:2], aurocs[:, 2:])] , f" - {name}")
            elif dataset=="FMNIST" or dataset=="CelebA":
                #print([f"{aa.mean():.2f} ({aa.std(axis=0).mean():.2f})" for aa in (aurocs[:, :1], aurocs[:, 1:])] , f" - {name}")
                print([f"{aa.mean():.3f} ({aa.std(axis=0).mean():.2f})" for aa in (aurocs[:, :1], aurocs[:, 1:2], aurocs[:, 2:3], aurocs[:, 3:4])] , f" - {name}")
            else:
                print(f"{aurocs.mean():.4f} ({aurocs.std(axis=0).mean():.2f})" , f" - {name}")
            #print(f"{label} - {np.mean(aurocs.mean(axis=0)):.3f} (std {np.mean(aurocs.std(axis=0)):.3f}) - {ood_datasets[0]}")
            c += 1
            #for d, ood_dataset in enumerate(ood_datasets[:2]):
            #for d, ood_dataset in enumerate(ood_datasets):
            #    print(f"\t\t {aurocs.mean(axis=0)[d]:.3f} (std {aurocs.std(axis=0)[d]:.3f}) - {ood_dataset}")




        #plt.title(f"{dataset} vs all - fixed memory budget of 3 nn")
        #plt.legend()
        #plt.legend(loc="upper right", edgecolor="black")
        plt.legend(loc="lower right", edgecolor="black")
        if dataset=="CIFAR-10":
            plt.ylim([0.45, 0.8])
        elif dataset=="MNIST":
            plt.ylim([0.2, 1.])
        elif dataset=="FMNIST":
            plt.ylim([0.56, 0.95])
            if budget==10:
                plt.ylim([0.56, 1.])
        #elif dataset=="CelebA":
        #    plt.ylim([0.51, 0.63])
        #else:
        #    plt.ylim([0.48,0.65])
        if dataset=="CIFAR-10" or dataset=="FMNIST" or dataset=="CelebA":
            ax.set_xticks(list(range(len(ood_datasets))), ood_datasets_xlabel, fontsize='medium')
        else:
            ax.set_xticks(list(range(len(ood_datasets))), ood_datasets, fontsize='medium')
        if dataset!="CelebA":
            plt.xticks(rotation = 80)
        plt.ylabel("AUROC", fontsize='medium')
        plt.grid(which="major", axis="both", linestyle="dotted")

        plt.savefig(f"figures/{dataset}_budget{budget}.pdf", bbox_inches='tight')
