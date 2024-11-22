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

for seed in {1..3}
do
    echo "#####################################################################################"
    echo "Seed $seed"
    python train_model.py --run_name warmup --dataset ImageNet --model SWIN_large --likelihood classification --n_epochs 10 --batch_size 128 --optimizer adamw --learning_rate 1e-6 --weight_decay 1e-3 --seed $seed
    python train_model.py --run_name e100 --dataset ImageNet --model SWIN_large --likelihood classification --n_epochs 100 --batch_size 128 --optimizer adamw --learning_rate 1e-4 --weight_decay 1e-3 --seed $seed --run_name_pretrained warmup
    python train_model.py --run_name e200 --dataset ImageNet --model SWIN_large --likelihood classification --n_epochs 100 --batch_size 128 --optimizer adamw --learning_rate 1e-5 --weight_decay 1e-3 --seed $seed --run_name_pretrained e100
    python train_model.py --run_name good --dataset ImageNet --model SWIN_large --likelihood classification --n_epochs 100 --batch_size 128 --optimizer adamw --learning_rate 1e-6 --weight_decay 1e-3 --seed $seed --run_name_pretrained e200
    python test_model.py --run_name good --dataset ImageNet --model SWIN_large --batch_size 128 --seed $seed 
done    


# this takes around 3 hours (per epoch and per seed) on a single H100 GPU