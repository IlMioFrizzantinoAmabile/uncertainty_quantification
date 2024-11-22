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

dataset="FMNIST"
ood_datasets="MNIST FMNIST-R"

model_name="LeNet"
run_name="good"

echo "Computing Projected Ensemble on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"
model_seeds=(1 2 3)

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "Projected Ensemble - memory budget 3p"
ensemble_sizes=(3)
for model_seed in "${model_seeds[@]}"
do
    for ensemble_size in "${ensemble_sizes[@]}"
    do
        echo "#######################################################################################################"
        echo " - model seed $model_seed - lanczos seed $lanczos_seed"
        python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --score projected_ensemble --ensemble_size $ensemble_size --subsample_trainset 60000 --train_batch_size 32 --test_batch_size 128 --n_epochs_projected_ensemble 1
    done
done

python score_model.py --ID_dataset MNIST --OOD_datasets FMNIST MNIST-R --model MLP_depth1_hidden20 --model_seed 1 --run_name good --score projected_ensemble --ensemble_size 5 --subsample_trainset 60000 --train_batch_size 32 --test_batch_size 128 --n_epochs_projected_ensemble 1
