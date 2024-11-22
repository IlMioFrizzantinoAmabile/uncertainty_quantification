#!/bin/bash
#BSUB -J scoring_pr_cifar
#BSUB -q p1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/scoring%J.out
#BSUB -e logs/scoring%J.err

source virtualenv/bin/activate

dataset="CIFAR-10"
ood_datasets="SVHN CIFAR-100"

model_name="ResNet"
run_name="good"

echo "Computing Projected Ensemble on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"
model_seeds=(1 2 3)

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "Projected Ensemble - memory budget 10p"
ensemble_sizes=(300 1000)
epochs=(1 3 10)

for epoch in "${epochs[@]}"
do
    for model_seed in "${model_seeds[@]}"
    do
        for ensemble_size in "${ensemble_sizes[@]}"
        do
            echo "#######################################################################################################"
            echo " - model seed $model_seed - lanczos seed $lanczos_seed"
            python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --score projected_ensemble --ensemble_size $ensemble_size --subsample_trainset 1000 --n_epochs_projected_ensemble $epoch --train_batch_size 32 --test_batch_size 16
        done
    done
done

#python score_model.py --ID_dataset CIFAR-10 --OOD_datasets SVHN CIFAR-100 CIFAR-10-C --model ResNet --model_seed 1 --run_name good --score projected_ensemble --ensemble_size 3 --subsample_trainset 50000 --train_batch_size 10 --test_batch_size 128  --n_epochs_projected_ensemble 1
#python score_model.py --ID_dataset CIFAR-10 --OOD_datasets SVHN CIFAR-100 --model ResNet --model_seed 1 --run_name good --score projected_ensemble --ensemble_size 10 --subsample_trainset 10000 --train_batch_size 32 --test_batch_size 128  --n_epochs_projected_ensemble 1
python score_model.py --ID_dataset CIFAR-10 --OOD_datasets SVHN CIFAR-100 --model ResNet --model_seed 1 --run_name good --score max_logit