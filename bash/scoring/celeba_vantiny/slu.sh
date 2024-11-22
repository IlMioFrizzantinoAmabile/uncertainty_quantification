#!/bin/bash
#BSUB -J scoring_slu
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
run_name="e50lr3"

echo "Computing SLU on $model_name trained on $dataset (run name $run_name) vs OOD datasets: $ood_datasets"
model_seeds=(1 2 3)
lanczos_seeds=(1)
sketch_seeds=(1)

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "SLU - memory budget 3p"
lanczos_lm_iter="100"
sketch_size="10000"
for model_seed in "${model_seeds[@]}"
do
    for lanczos_seed in "${lanczos_seeds[@]}"
    do
        for sketch_seed in "${sketch_seeds[@]}"
        do
            echo "#######################################################################################################"
            echo " - model seed $model_seed - lanczos seed $lanczos_seed"
            python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --lanczos_hm_iter 0 --lanczos_lm_iter $lanczos_lm_iter --lanczos_seed $lanczos_seed --sketch srft --sketch_size $sketch_size --sketch_seed $sketch_seed --subsample_trainset 10000 --test_batch_size 8 --train_batch_size 32 --serialize_ggn_on_batches
        done
    done
done

echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "#######################################################################################################################################"
echo "SLU - memory budget 10p"
lanczos_lm_iter="200"
sketch_size="10000"
for model_seed in "${model_seeds[@]}"
do
    for lanczos_seed in "${lanczos_seeds[@]}"
    do
        for sketch_seed in "${sketch_seeds[@]}"
        do
            echo "#######################################################################################################"
            echo " - model seed $model_seed - lanczos seed $lanczos_seed"
            python score_model.py --ID_dataset $dataset --OOD_datasets $ood_datasets --model $model_name --model_seed $model_seed --run_name $run_name --lanczos_hm_iter 0 --lanczos_lm_iter $lanczos_lm_iter --lanczos_seed $lanczos_seed --sketch srft --sketch_size $sketch_size --sketch_seed $sketch_seed --subsample_trainset 10000 --test_batch_size 8 --train_batch_size 32 --serialize_ggn_on_batches
        done
    done
done