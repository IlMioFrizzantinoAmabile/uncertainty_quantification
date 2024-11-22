bash/setup.sh
source ./virtualenv/bin/activate

bash/train/celeba_vantiny.sh

bash/scoring/celeba_vantiny/deep_ensemble.sh
bash/scoring/celeba_vantiny/laplace_diagonal.sh
bash/scoring/celeba_vantiny/laplace_lowrank.sh
bash/scoring/celeba_vantiny/local_ensemble.sh
bash/scoring/celeba_vantiny/local_ensemblehessian.sh
bash/scoring/celeba_vantiny/scod.sh
bash/scoring/celeba_vantiny/slu.sh
bash/scoring/celeba_vantiny/swag.sh

python plot_fixed_memory_budget.py --dataset CelebA --budget 3
python plot_fixed_memory_budget.py --dataset CelebA --budget 10