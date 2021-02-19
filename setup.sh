sudo apt update
sudo apt install cmake emacs build-essential unzip

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p ./miniconda
source "./miniconda/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -y -n test_env python=3.8
conda activate test_env
conda install --yes --file requirements.txt

source setup_ext.sh
