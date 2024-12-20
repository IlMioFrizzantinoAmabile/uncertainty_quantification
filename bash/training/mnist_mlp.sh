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
    python train_model.py --run_name good --dataset MNIST --model MLP --mlp_num_layers 1 --mlp_hidden_dim 20 --activation_fun tanh --likelihood classification --n_epochs 50 --batch_size 128 --optimizer adam --learning_rate 1e-3 --seed $seed
    python test_model.py --run_name good --dataset MNIST --model MLP_depth1_hidden20 --batch_size 128 --seed $seed
done

# this takes around 1 minute (per seed) on a single H100 GPU