#!/bin/bash
#BSUB -J train_celeba
#BSUB -q p1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 72:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/train%J.out
#BSUB -e logs/train%J.err

source ./virtualenv/bin/activate

for seed in {1..10}
do
    echo "#####################################################################################"
    echo "Seed $seed"
    python train_model.py --run_name good --dataset CelebA --model VAN_tiny --likelihood binary_multiclassification --n_epochs 50 --batch_size 128 --optimizer adam --learning_rate 1e-3 --decrease_learning_rate --seed $seed --test_every_n_epoch 10
    python test_model.py --run_name good --dataset CelebA --model VAN_tiny --batch_size 128 --seed $seed 
done

# this takes around 10 hours (per seed) on a single H100 GPU