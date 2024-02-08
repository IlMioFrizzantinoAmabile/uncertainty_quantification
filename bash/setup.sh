echo
module load python3/3.9.11 cuda/11.4 cudnn/v8.6.0.163-prod-cuda-11.X
python3 -m pip install --upgrade pip
python3 -m venv virtualenv
source ./virtualenv/bin/activate
pip install -r requirements.txt
# ACHTUNG! The next line of code installs jax==0.4.6
# thus downgrading jax==0.4.23.
python3 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# I make sure to re-install jax==0.4.23 because
# after jax[cuda11_pip] is installed we are down to jax==0.4.6
pip install jax==0.4.23 jaxlib==0.4.23
