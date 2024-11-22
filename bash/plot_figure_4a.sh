bash/setup.sh
source ./virtualenv/bin/activate

bash/train/mnist_mlp.sh

bash/scoring/mnist_mlp/deep_ensemble.sh
bash/scoring/mnist_mlp/laplace_diagonal.sh
bash/scoring/mnist_mlp/laplace_lowrank.sh
bash/scoring/mnist_mlp/local_ensemble.sh
bash/scoring/mnist_mlp/local_ensemblehessian.sh
bash/scoring/mnist_mlp/scod.sh
bash/scoring/mnist_mlp/slu.sh
bash/scoring/mnist_mlp/swag.sh

python plot_fixed_memory_budget.py --dataset MNIST --budget 3
python plot_fixed_memory_budget.py --dataset MNIST --budget 10