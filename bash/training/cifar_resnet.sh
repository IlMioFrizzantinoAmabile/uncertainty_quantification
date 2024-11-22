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

source virtualenv/bin/activate

for seed in {1..10}
do
    echo "#####################################################################################"
    echo "Seed $seed"
    python train_model.py --run_name good --dataset CIFAR-10 --model ResNet --activation_fun relu --likelihood classification --n_epochs 200 --batch_size 128 --optimizer sgd --learning_rate 0.1 --decrease_learning_rate --momentum 0.9 --weight_decay 1e-4 --seed $seed
    python test_model.py --run_name good --dataset CIFAR-10 --model ResNet --batch_size 128 --seed $seed
done

