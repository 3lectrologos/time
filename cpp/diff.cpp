#include <iostream>
#include <algorithm>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "eigen/Eigen/Dense"


namespace py = pybind11;


typedef std::vector<std::vector<double>> mat_t;
typedef std::vector<double> seq_t;


template <typename T> class Nparr {
public:
  Nparr(py::array_t<T>& array) : buf(array.request()) {
    data_ = (T*) buf.ptr;
    shape = buf.shape;
  }
  
  size_t index(size_t i, size_t j) const {
    return (i * buf.strides[0] + j * buf.strides[1]) / buf.itemsize;
  }
  
  T operator ()(const size_t i, const size_t j) const {
    return data_[index(i, j)];
  }
  
  T& operator ()(size_t i, size_t j) {
    return data_[index(i, j)];
  }
  
  size_t index(size_t i) const {
    return (i * buf.strides[0]) / buf.itemsize;
  }
  
  T operator ()(size_t i) const {
    return data_[index(i)];
  }
  
  T& operator ()(size_t i) {
    return data_[index(i)];
  }
  
public:
  py::buffer_info buf;
  
public:
  T* data_;
  std::vector<long int> shape;
};


template <typename T> py::array_t<T> npmat(const size_t nrows, const size_t ncols) {
  auto result = py::array(py::buffer_info(
    nullptr,                            // Pointer to data (nullptr -> ask NumPy to allocate!)
    sizeof(T),                          // Size of one item
    py::format_descriptor<T>::format(), // Buffer format
    2,                                  // How many dimensions?
    { nrows, ncols },                   // Number of elements for each dimension
    { sizeof(T), nrows * sizeof(T) }    // Strides for each dimension
  ));
  
  return result;
}


double logaddexp(const double a, const double b) {
  if (a == -std::numeric_limits<double>::infinity()) {
    return b;
  }
  else if (b == -std::numeric_limits<double>::infinity()) {
    return a;
  }
  else {
    double max_exp = std::max(a, b);
    double sum = exp(a - max_exp) + exp(b - max_exp);
    return log(sum) + max_exp;
  }
}


double logsumexp(const std::vector<double>& a) {
  double max_exp = a[0];
  for (auto const& v : a) {
    if (v > max_exp) {
      max_exp = v;
    }
  }
  
  double sum = 0.0;
  for (auto const& v : a) {
    sum += exp(v - max_exp);
  }
  
  return log(sum) + max_exp;
}


std::pair<double, Eigen::MatrixXd> loglik_seq(py::array_t<double> theta, const seq_t& seq) {
  Nparr<double> th(theta);
  int n = th.shape[0];
  seq_t ground(n);
  std::iota(ground.begin(), ground.end(), 0);
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);

  double lik = 0;
  for (auto k=0; k < seq.size()+1; k++) {
    if (k < seq.size()) {
      int i = seq[k];
      lik += th(i, i);
      grad(i, i) += 1;
      for(auto r=k+1; r < seq.size(); r++) {
        int j = seq[r];
        lik += th(i, j);
        grad(i, j) += 1;
      }
    }

    seq_t rest;
    seq_t sortedseq(seq.begin(), seq.begin()+k);
    std::sort(sortedseq.begin(), sortedseq.end());
    std::set_difference(ground.begin(), ground.end(),
                        sortedseq.begin(), sortedseq.end(),
                        std::inserter(rest, rest.begin()));

    seq_t sumth(rest.size());
    for (auto r=0; r < rest.size(); r++) {
      auto j = rest[r];
      sumth[r] = th(j, j);
      for (auto s=0; s < k; s++) {
        auto i = seq[s];
        sumth[r] += th(i, j);
      }
    }
    double lse = logaddexp(0, logsumexp(sumth));
    lik -= lse;
    for (auto r=0; r < rest.size(); r++) {
      auto j = rest[r];
      grad(j, j) -= exp(sumth[r] - lse);
      for (auto s=0; s < k; s++) {
        auto i = seq[s];
        grad(i, j) -= exp(sumth[r] - lse);
      }
    }
  }

  return std::make_pair(lik, grad);
}


std::pair<double, Eigen::MatrixXd> loglik_set(py::array_t<double> theta, const seq_t& seq) {
  Nparr<double> th(theta);
  int n = th.shape[0];
  long nperm = 1;
  for (auto i = 2; i < seq.size()+1; i++) {
    nperm *= i;
  }
  
  std::vector<double> liks(nperm);
  std::vector<Eigen::MatrixXd> grads(nperm);

  seq_t pseq(seq);
  std::sort(pseq.begin(), pseq.end());
  unsigned int i = 0;
  do {
    auto res = loglik_seq(theta, pseq);
    liks[i] = res.first;
    grads[i] = res.second;
    i++;
  } while (std::next_permutation(pseq.begin(), pseq.end()));

  double lse = logsumexp(liks);
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);
  for (auto i=0; i < nperm; i++) {
    grad += exp(liks[i] - lse) * grads[i];
  }

  return std::make_pair(lse, grad);
}


PYBIND11_MODULE(diff, m) {
  m.def("loglik_seq", &loglik_seq, py::return_value_policy::reference_internal);
  m.def("loglik_set", &loglik_set, py::return_value_policy::reference_internal);
}
