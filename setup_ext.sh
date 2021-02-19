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
