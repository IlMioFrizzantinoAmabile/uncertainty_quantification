#!/bin/bash
#BSUB -J scoring_swag
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

dataset="ImageNet"
ood_datasets="ImageNet-classout"

model_name="SWIN_large"
run_name="good"

echo "Computing SWAG on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"
model_seeds=(1)

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "SWAG - memory budget 3p"
n_vec="3"
for model_seed in "${model_seeds[@]}"
do
    echo "#######################################################################################################"
    echo " - model seed $model_seed"
    python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --score swag --swag_n_vec $n_vec --swag_lr 1e-5 --swag_momentum 0.99 --swag_collect_interval 1000 --train_batch_size 128 --test_batch_size 8
done