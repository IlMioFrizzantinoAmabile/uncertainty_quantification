bash/setup.sh
source ./virtualenv/bin/activate

bash/train/cifar_resnet.sh

bash/scoring/cifar_resnet/deep_ensemble.sh
bash/scoring/cifar_resnet/laplace_diagonal.sh
bash/scoring/cifar_resnet/laplace_lowrank.sh
bash/scoring/cifar_resnet/local_ensemble.sh
bash/scoring/cifar_resnet/local_ensemblehessian.sh
bash/scoring/cifar_resnet/scod.sh
bash/scoring/cifar_resnet/slu.sh
bash/scoring/cifar_resnet/swag.sh

python plot_fixed_memory_budget.py --dataset CIFAR-10 --budget 3
python plot_fixed_memory_budget.py --dataset CIFAR-10 --budget 10