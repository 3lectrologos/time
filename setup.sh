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

cd cpp
git clone https://github.com/pybind/pybind11.git
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip -O eigen.zip
unzip -q eigen.zip
rm eigen.zip
mv eigen-3.3.9 eigen

cd cpp
sudo apt install cmake
cmake .
make
cd ../