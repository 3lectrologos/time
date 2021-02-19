#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <omp.h>
#include "eigen/Eigen/Dense"


namespace py = pybind11;


typedef std::vector<double> seq_t;


void print_seq(const seq_t& seq) {
  for (auto s : seq) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
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


double loglik_seq_nograd(const Eigen::MatrixXd& theta, const seq_t& seq) {
  int n = theta.rows();
  std::list<int> rest;
  for (auto i=0; i < n; i++) rest.push_back(i);

  double lik = 0;
  for (auto k=0; k < seq.size()+1; k++) {
    if (k < seq.size()) {
      int i = seq[k];
      lik += theta(i, i);
      for(auto r=k+1; r < seq.size(); r++) {
        int j = seq[r];
        lik += theta(i, j);
      }
    }

    if (rest.size() > 0) {
      seq_t sumth(rest.size());
      int r = 0;
      for (auto j : rest) {
        sumth[r] = theta(j, j);
        for (auto s=0; s < k; s++) {
          sumth[r] += theta(seq[s], j);
        }
        r++;
      }
      double lse = logaddexp(0, logsumexp(sumth));
      lik -= lse;
    }

    if (k < seq.size()) {
      auto it = std::find(rest.begin(), rest.end(), seq[k]);
      rest.erase(it);
    }
  }

  return lik;
}


std::pair<double, Eigen::MatrixXd> loglik_seq(const Eigen::MatrixXd& theta, const seq_t& seq) {
  int n = theta.rows();
  std::list<int> rest;
  for (auto i=0; i < n; i++) rest.push_back(i);
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);

  double lik = 0;
  for (auto k=0; k < seq.size()+1; k++) {
    if (k < seq.size()) {
      int i = seq[k];
      lik += theta(i, i);
      grad(i, i) += 1;
      for(auto r=k+1; r < seq.size(); r++) {
        int j = seq[r];
        lik += theta(i, j);
        grad(i, j) += 1;
      }
    }

    if (rest.size() > 0) {
      seq_t sumth(rest.size());
      int r = 0;
      for (auto j : rest) {
        sumth[r] = theta(j, j);
        for (auto s=0; s < k; s++) {
          sumth[r] += theta(seq[s], j);
        }
        r++;
      }

      double lse = logaddexp(0, logsumexp(sumth));
      lik -= lse;
      r = 0;
      for (auto j : rest) {
        grad(j, j) -= exp(sumth[r] - lse);
        for (auto s=0; s < k; s++) {
          grad(seq[s], j) -= exp(sumth[r] - lse);
        }
        r++;
      }
    }
    
    if (k < seq.size()) {
      auto it = std::find(rest.begin(), rest.end(), seq[k]);
      rest.erase(it);
    }
  }

  return std::make_pair(lik, grad);
}


std::vector<seq_t> all_perms(const seq_t& set) {
  std::vector<seq_t> result;
  seq_t sorted = seq_t(set);
  std::sort(sorted.begin(), sorted.end());
  do {
    result.push_back(sorted);
  } while(std::next_permutation(sorted.begin(), sorted.end()));
  return result;
}


std::vector<seq_t> sample_perms(const Eigen::MatrixXd& theta, const seq_t& set, int nperms) {
  int burnin = nperms;
  std::vector<seq_t> result;
  seq_t cur(set);
  auto pcur = loglik_seq_nograd(theta, cur);
  for(auto i=0; i < burnin+nperms; i++) {
    seq_t next(cur);
    std::random_shuffle(next.begin(), next.end());
    auto pnext = loglik_seq_nograd(theta, next);
    auto paccept = std::min(1.0, pnext/pcur);
    if ((double) rand() / (RAND_MAX) > paccept) {
      pcur = pnext;
    }
    if (i >= burnin) {
      result.push_back(next);
    }
  }
  return result;
}


std::pair<double, Eigen::MatrixXd> loglik_set_aux(
    const Eigen::MatrixXd& theta, const seq_t& set, std::vector<seq_t> perms) {
  auto n = theta.rows();
  std::vector<double> liks(perms.size());
  std::vector<Eigen::MatrixXd> grads(perms.size());

  for (auto i=0; i < perms.size(); i++) {
    auto res = loglik_seq(theta, perms[i]);
    liks[i] = res.first;
    grads[i] = res.second;
  }
  
  double lse = logsumexp(liks);

  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);
  for (auto i=0; i < perms.size(); i++) {
    grad += exp(liks[i] - lse) * grads[i];
  }

  return std::make_pair(lse, grad);
}


std::pair<double, Eigen::MatrixXd> loglik_set_full(const Eigen::MatrixXd& theta, const seq_t& set) {
  auto perms = all_perms(set);
  return loglik_set_aux(theta, set, perms);
}


std::pair<double, Eigen::MatrixXd> loglik_set(const Eigen::MatrixXd& theta, const seq_t& set, int nperms) {
  auto perms = sample_perms(theta, set, nperms);
  return loglik_set_aux(theta, set, perms);
}


// TODO: Refactor this and the above?
std::pair<double, Eigen::MatrixXd> loglik_data_full(const Eigen::MatrixXd& theta, const std::vector<seq_t>& data) {
  auto n = theta.rows();
  double lik = 0;
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);
#pragma omp parallel for
  for (auto i=0; i < data.size(); i++) {
    auto res = loglik_set_full(theta, data[i]);
    lik += res.first;
    grad += res.second;
  }
  lik /= data.size();
  grad /= data.size();
  return std::make_pair(lik, grad);
}


Eigen::MatrixXd loglik_data(const Eigen::MatrixXd& theta, const std::vector<seq_t>& data, int nperms) {
  auto n = theta.rows();
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);
#pragma omp parallel for
  for (auto i=0; i < data.size(); i++) {
    // TODO: Refactor this
    if (data[i].size() < 5) {
      grad += loglik_set_full(theta, data[i]).second;
    } else {
      grad += loglik_set(theta, data[i], nperms).second;
    }
  }
  grad /= data.size();
  return grad;
}


PYBIND11_MODULE(diff, m) {
  m.def("loglik_seq", &loglik_seq, py::return_value_policy::reference_internal);
  m.def("loglik_set", &loglik_set, py::return_value_policy::reference_internal);
  m.def("loglik_set_full", &loglik_set_full, py::return_value_policy::reference_internal);
  m.def("loglik_data", &loglik_data, py::return_value_policy::reference_internal);
  m.def("loglik_data_full", &loglik_data_full, py::return_value_policy::reference_internal);
}


int main() {
  int n = 20;
  Eigen::MatrixXd theta = Eigen::MatrixXd::Zero(n, n);
  seq_t set = {2, 0, 1, 3, 5, 8};
  double step = 0.01;

  for (auto i=0; i < 100; i++) {
    auto res = loglik_set(theta, set, 100);
    theta += step*res.second;
    std::cout << res.first << std::endl;
  }
}
