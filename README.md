# Uncertainty Quantification

This repo is a benchmark for Bayesian neural network methods for quantifying uncertainty

## Setup 
Clone the repository using the command `Git: Clone` and the URL of the repo.
- Open the terminal and run the following command:
```bash
sh bash/setup.sh
```

## Reproduce results

#### Train model

Train a single model with
```
source ./virtualenv/bin/activate;
CUDA_VISIBLE_DEVICES=0;
python train_model.py --dataset MNIST --likelihood "classification" --model LeNet
```

or train all the models (on a cluster) with
```
bash/train_all.sh
```

#### Compute OOD metric

Compute the scores for a single model with
```
source ./virtualenv/bin/activate;
CUDA_VISIBLE_DEVICES=0;
python compute_ood_score.py --ID_dataset MNIST --OOD_dataset FMNIST --model_name LeNet --score local_ensemble --run_name "run_name" --seed 420 --subsample 1
```

or compute scores for all the models (on a cluster) with
```
bash/score_all.sh
```