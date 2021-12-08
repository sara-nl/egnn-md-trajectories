module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/11.0.2-GCC-9.3.0

VIRTENV=EGNN
VIRTENV_ROOT=~/.virtualenvs

deactivate
conda deactivate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
  yes | rm -r $VIRTENV_ROOT/$VIRTENV
  python3 -m venv $VIRTENV_ROOT/$VIRTENV
fi

source $VIRTENV_ROOT/$VIRTENV/bin/activate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  pip install --upgrade pip
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
  pip install tqdm --no-cache-dir
  pip install tensorboardX --no-cache-dir
  pip install wheel --no-cache-dir
  pip install ninja --no-cache-dir
  pip install egnn-pytorch --no-cache-dir
fi