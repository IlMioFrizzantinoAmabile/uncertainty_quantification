#!/bin/bash
#BSUB -J scoring
#BSUB -q p1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/scoring%J.out
#BSUB -e logs/scoring%J.err

source ./virtualenv/bin/activate

dataset="Sinusoidal"
ood_datasets="Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace"

model_name="MLP_depth2_hidden50"
run_name="good"

# projected ensemble
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --score projected_ensemble --use_proj_loss --ensemble_size 100 --n_epochs_projected_ensemble 100 --train_batch_size 1 --test_batch_size 4

# diagonal laplace
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --score diagonal_lla --hutchinson_samples 2000 --test_batch_size 5 --train_batch_size 10 --serialize_ggn_on_batches --verbose

# linearized laplace
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --lanczos_hm_iter 20 --lanczos_lm_iter 0 --lanczos_seed 1 --test_batch_size 5 --use_eigenval --train_batch_size 10 --serialize_ggn_on_batches

# local ensemble
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --lanczos_hm_iter 20 --lanczos_lm_iter 0 --lanczos_seed 1 --test_batch_size 5 --train_batch_size 10 --serialize_ggn_on_batches
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --lanczos_hm_iter 3 --lanczos_lm_iter 0 --lanczos_seed 1 --test_batch_size 5 --train_batch_size 10 --serialize_ggn_on_batches

# local ensemble hessian
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --lanczos_hm_iter 20 --lanczos_lm_iter 0 --lanczos_seed 1 --test_batch_size 5 --use_hessian --train_batch_size 10 --serialize_ggn_on_batches

# sketched laplace
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --lanczos_hm_iter 0 --lanczos_lm_iter 20 --lanczos_seed 1 --sketch srft --sketch_size 2000 --sketch_seed 1 --test_batch_size 5 --train_batch_size 10 --serialize_ggn_on_batches 
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good --lanczos_hm_iter 0 --lanczos_lm_iter 20 --lanczos_seed 1 --sketch srft --sketch_size 405 --sketch_seed 1 --test_batch_size 5 --train_batch_size 10 --serialize_ggn_on_batches 

# swag
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good  --score swag --swag_n_vec 20 --swag_lr 1e-3 --swag_momentum 0.99 --swag_collect_interval 100 --train_batch_size 2 --test_batch_size 5 --verbose
python score_model.py --ID_dataset Sinusoidal --OOD_datasets Sinusoidal-left Sinusoidal-center Sinusoidal-right Sinusoidal-linspace --model MLP_depth2_hidden50 --model_seed 420 --run_name good  --score swag --swag_n_vec 20 --swag_lr 1e-3 --swag_momentum 0.9 --swag_collect_interval 100 --train_batch_size 2 --test_batch_size 5 --verbose