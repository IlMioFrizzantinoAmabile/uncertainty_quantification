# Uncertainty Quantification

This repo is a benchmark for Bayesian neural network methods for quantifying uncertainty.

It supports several datasets ("MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "ImageNet") and several model architectutres ("MLP", "LeNet", "GoogleNet", "ConvNeXt", "ResNet", "VAN", "SWIN"). Virtually any combination is possible. After a model is trained we can compute several Uncertainty Quantification scores ("scod", "swag", "ensemble", "local_ensemble", "sketched_local_ensemble", "low_rank_lla", "diagonal_lla") against a series of OoD datasets.

## Setup 
Clone the repository using the command `Git: Clone` and the URL of the repo.
- Open the terminal and run the following command:
```bash
bash/setup.sh
```
This will create a virtual environment and install all the required packages. After the first use, you will only need to activate the environment by calling

```bash
source virtualenv/bin/activate
```

## Reproduce results

#### Train model

Train any combination of model and dataset with (for example)
```
source ./virtualenv/bin/activate;
CUDA_VISIBLE_DEVICES=0;

python train_model.py --dataset MNIST --likelihood classification --model MLP --default_hyperparams 

python train_model.py --dataset FMNIST --likelihood classification --model LeNet --default_hyperparams

python train_model.py --dataset CIFAR-10 --likelihood classification --model ResNet --default_hyperparams

python train_model.py --dataset CelebA --likelihood binary_multiclassification --model VAN_tiny --default_hyperparams

python train_model.py --dataset ImageNet --likelihood classification --model SWIN_large --default_hyperparams
```

#### Compute OOD metric

Compute the scores for a single model with (for example)
```
source ./virtualenv/bin/activate;
CUDA_VISIBLE_DEVICES=0;

python score_model.py --ID_dataset FMNIST --OOD_dataset MNIST FMNIST-R --model_name LeNet --score local_ensemble

python score_model.py --ID_dataset CIFAR-10 --OOD_dataset SVHN CIFAR-10-C --model_name ResNet --score scod

python score_model.py --ID_dataset CelebA --OOD_dataset FOOD101 CelebA-Mustache CelebA-Bald CelebA-Eyeglasses --model_name VAN_tiny --subsample_trainset 10000 --lanczos_hm_iter 3 --lanczos_lm_iter 0 --test_batch_size 8 --train_batch_size 32 --serialize_ggn_on_batches

python score_model.py --ID_dataset ImageNet --OOD_datasets SVHN-256 FOOD101-256 ImageNet-classout --model VAN_large --subsample_trainset 100000 --lanczos_hm_iter 0 --lanczos_lm_iter 10 --test_batch_size 8 --train_batch_size 32 --serialize_ggn_on_batches --sketch srft --sketch_size 10000000
```