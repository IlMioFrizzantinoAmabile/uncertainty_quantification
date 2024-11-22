bash/setup.sh
source ./virtualenv/bin/activate

bash/train/imagenet_swin.sh

bash/scoring/imagenet_swin/laplace_diagonal.sh
bash/scoring/imagenet_swin/laplace_lowrank.sh
bash/scoring/imagenet_swin/local_ensemble.sh
bash/scoring/imagenet_swin/local_ensemblehessian.sh
bash/scoring/imagenet_swin/slu.sh
bash/scoring/imagenet_swin/swag.sh

python plot_fixed_memory_budget.py --dataset ImageNet --budget 3