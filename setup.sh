cd cpp
git clone https://github.com/pybind/pybind11.git
curl -O https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
git clone https://github.com/autodiff/autodiff
mv autodiff autodiff_tmp
mv autodiff_tmp/autodiff .
rm -rf autodiff_tmp
tar -xf eigen-3.3.9.tar.gz
rm eigen-3.3.9.tar.gz
mv eigen-3.3.9 eigen

cmake .
make
cd ../
