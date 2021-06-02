Requirements
------------
* CMake, unzip, C++ compiler (e.g., `sudo apt install cmake unzip build-essential`)
* Python 3.7 or later with the following libraries: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`

Install
-------
Running `setup.sh` downloads the required third-party C++ libraries (`pybind11`, `autodiff`, `eigen`), and builds the C++ code.

Run
---
* `python learn.py` to learn a model on the top 50 genes of the TCGA GBM data
* `run_synthetic.sh` to run the synthetic experiments of the paper
* `run_real.sh` to run the real data experiments of the paper