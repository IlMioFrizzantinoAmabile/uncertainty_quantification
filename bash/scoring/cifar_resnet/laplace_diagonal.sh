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

dataset="CIFAR-10"
ood_datasets="SVHN CIFAR-100"

model_name="ResNet"
run_name="good"

echo "Computing Diagonal Laplace on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "Diagonal Laplace - memory budget 1p"
model_seeds=(1 2 3)
for model_seed in "${model_seeds[@]}"
do  
    echo "Model seed: $model_seed"
    python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --score diagonal_lla  --subsample_trainset 50000 --test_batch_size 128 --train_batch_size 128 --serialize_ggn_on_batches
done