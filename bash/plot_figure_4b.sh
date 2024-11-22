bash/setup.sh
source ./virtualenv/bin/activate

bash/train/fmnist_lenet.sh

bash/scoring/fmnist_lenet/deep_ensemble.sh
bash/scoring/fmnist_lenet/laplace_diagonal.sh
bash/scoring/fmnist_lenet/laplace_lowrank.sh
bash/scoring/fmnist_lenet/local_ensemble.sh
bash/scoring/fmnist_lenet/local_ensemblehessian.sh
bash/scoring/fmnist_lenet/scod.sh
bash/scoring/fmnist_lenet/slu.sh
bash/scoring/fmnist_lenet/swag.sh

python plot_fixed_memory_budget.py --dataset FMNIST --budget 3
python plot_fixed_memory_budget.py --dataset FMNIST --budget 10