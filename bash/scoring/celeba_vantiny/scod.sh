#!/bin/bash
#BSUB -J scoring_scod
#BSUB -q p1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 72:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/scoring%J.out
#BSUB -e logs/scoring%J.err

source virtualenv/bin/activate

dataset="CelebA"
ood_datasets="FOOD101 CelebA-classout"

model_name="VAN_tiny"
run_name="good"

echo "Computing SCOD on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"
model_seeds=(1 2 3)

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "SCOD - memory budget 3p"
n_vec="3"
for model_seed in "${model_seeds[@]}"
do
    echo "#######################################################################################################"
    echo " - model seed $model_seed"
    python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --score scod --n_eigenvec_hm $n_vec --subsample_trainset 10000 --test_batch_size 8
done


echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "SCOD - memory budget 10p"
n_vec="10"
for model_seed in "${model_seeds[@]}"
do
    echo "#######################################################################################################"
    echo " - model seed $model_seed"
    python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --score scod --n_eigenvec_hm $n_vec --subsample_trainset 10000 --test_batch_size 8
done
