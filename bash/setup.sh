echo

module load python3/3.12.4 cuda/12.4 cudnn/v8.9.7.29-prod-cuda-12.X 
python3 -m venv virtualenv
source virtualenv/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install --upgrade "jax[cuda12_pip]"==0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements_noversion.txt
python3 -m pip install --upgrade "jax[cuda12_pip]"==0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html