echo
module load python3/3.12.4 cuda/12.4 cudnn/v8.9.7.29-prod-cuda-12.X
python3 -m venv virtualenv
source virtualenv/bin/activate
python3 -m pip install --upgrade pip
pip install -U "jax[cuda12]"
pip install -r requirements_noversion.txt