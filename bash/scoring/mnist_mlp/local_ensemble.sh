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

dataset="MNIST"
ood_datasets="FMNIST KMNIST MNIST-R"

model_name="MLP_depth1_hidden20"
run_name="good"

echo "Computing Local Ensemble on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"
model_seeds=(1 2 3)
lanczos_seeds=(1 2 3)

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "Local Ensemble - memory budget 3p"
lanczos_hm_iter="3"
for model_seed in "${model_seeds[@]}"
do
    for lanczos_seed in "${lanczos_seeds[@]}"
    do
        echo "#######################################################################################################"
        echo " - model seed $model_seed - lanczos seed $lanczos_seed"
        python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --lanczos_hm_iter $lanczos_hm_iter --lanczos_lm_iter 0 --lanczos_seed $lanczos_seed --subsample_trainset 50000 --test_batch_size 128
    done
done

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "Local Ensemble - memory budget 10p"
lanczos_hm_iter="10"
for model_seed in "${model_seeds[@]}"
do
    for lanczos_seed in "${lanczos_seeds[@]}"
    do
        echo "#######################################################################################################"
        echo " - model seed $model_seed - lanczos seed $lanczos_seed"
        python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --lanczos_hm_iter $lanczos_hm_iter --lanczos_lm_iter 0 --lanczos_seed $lanczos_seed --subsample_trainset 50000 --test_batch_size 128
    done
done