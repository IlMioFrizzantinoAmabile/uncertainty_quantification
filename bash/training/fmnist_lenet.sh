#!/bin/bash
#BSUB -J train
#BSUB -q p1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/train%J.out
#BSUB -e logs/train%J.err

source ./virtualenv/bin/activate

for seed in {1..10}
do
    echo "#####################################################################################"
    echo "Seed $seed"
    python train_model.py --run_name good --dataset FMNIST --model LeNet --activation_fun tanh --likelihood classification --n_epochs 50 --batch_size 128 --optimizer adam --learning_rate 1e-3 --seed $seed
    python test_model.py --run_name good --dataset FMNIST --model LeNet --batch_size 128 --seed $seed
done