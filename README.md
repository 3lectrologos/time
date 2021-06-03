Requirements
------------
* CMake 3.9 or later, C++ compiler with OpenMP support
* Python 3.7 or later with the following libraries: `numpy`, `scipy`, `matplotlib`, `pandas`

Install
-------
Running `setup.sh` downloads the required third-party C++ libraries (`pybind11`, `autodiff`, `eigen`), and builds the C++ code.

Run
---
* `python learn.py` to learn a model on the top 50 genes of the TCGA GBM data
* `run_synthetic.sh [NUM_REPETITIONS]` to run the synthetic experiments. (In the paper we ran `run_synthetic.sh 96`.)
* `run_real.sh [NUM_REPETITIONS]` to run the real data experiments. (In the paper we ran `run_real.sh 24`.)