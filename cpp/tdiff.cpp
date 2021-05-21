#include <cmath>
#include "eigen/Eigen/Dense"
#include "autodiff/forward.hpp"
#include "autodiff/forward/eigen.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <omp.h>


using namespace autodiff;


typedef std::vector<int> seq_t;


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


dual p0(dual t1, dual t2, dual t12, dual t21, dual time) {
  auto h1 = exp(t1);
  auto h2 = exp(t2);
  auto h12 = exp(t12);
  auto h21 = exp(t21);
  auto q0 = h1 + h2;
  auto q1 = h21*h1;
  auto q2 = h12*h2;
  return -q0*time;
}


dual p1(dual t1, dual t2, dual t12, dual t21, dual time) {
  auto h1 = exp(t1);
  auto h2 = exp(t2);
  auto h12 = exp(t12);
  auto h21 = exp(t21);
  auto q0 = h1 + h2;
  auto q1 = h21*h1;
  auto q2 = h12*h2;
  return log(h1*(exp(-q2*time) - exp(-q0*time)) / (q0-q2));
}


dual p2(dual t1, dual t2, dual t12, dual t21, dual time) {
  auto h1 = exp(t1);
  auto h2 = exp(t2);
  auto h12 = exp(t12);
  auto h21 = exp(t21);
  auto q0 = h1 + h2;
  auto q1 = h21*h1;
  auto q2 = h12*h2;
  return log(h2*(exp(-q1*time) - exp(-q0*time)) / (q0-q1));
}


dual p12(dual t1, dual t2, dual t12, dual t21, dual time) {
  auto h1 = exp(t1);
  auto h2 = exp(t2);
  auto h12 = exp(t12);
  auto h21 = exp(t21);
  auto q0 = h1 + h2;
  auto q1 = h21*h1;
  auto q2 = h12*h2;
  auto res12 = (h1/(q0*(q0-q2)))*(q0*exp(-q2*time) - q2*exp(-q0*time));
  auto res21 = (h2/(q0*(q0-q1)))*(q0*exp(-q1*time) - q1*exp(-q0*time));
  return log(1 - res12 - res21);
  //return log(1 - exp(-q0*time) - (h1/(q0-q2))*(exp(-q2*time) - exp(-q0*time)) - (h2/(q0-q1))*(exp(-q1*time) - exp(-q0*time)));
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


dual hypoexp(const VectorXdual& qs, double time) {
  if (qs.size() == 0) {
    dual res = 1;
    return 1;
  }

  dual res = 0;
  for (auto i=0; i < qs.size(); i++) {
    if (qs(i) == 0) {
      return 0;
    }
    dual denom = 1;
    for (auto j=0; j < qs.size(); j++) {
      if (j != i) {
        denom *= (qs(j)-qs(i));
      }
    }
    res += (1-exp(-qs(i)*time)) / (qs(i)*denom);
  }
  res *= qs.prod();
  return res;
}


dual loglik_seq_aux(const VectorXdual& th, const seq_t& seq, double time, int n) {
  std::list<int> rest;
  for (auto i=0; i < n; i++) rest.push_back(i);

  dual lik = 0;
  VectorXdual qs1(seq.size());
  VectorXdual qs2(seq.size()+1);
  
  for (auto k=0; k < seq.size()+1; k++) {
    if (k < seq.size()) qs1(k) = 0;
    qs2(k) = 0;

    for (auto j : rest) {
      dual st = th(j+n*j);
      for (auto i=0; i < k; i++) {
        st += th(j+n*seq[i]);
      }
      if (k < seq.size()) {
        qs1(k) += exp(st);
        if (j == seq[k]) {
          lik += st;
        }
      }
      qs2(k) += exp(st);
    }

    if (k < seq.size()) {
      lik -= log(qs1(k));
      auto it = std::find(rest.begin(), rest.end(), seq[k]);
      rest.erase(it);
    }
  }

  // NOTE: The 1e-8 is added to avoid numerical issues.
  lik += log(hypoexp(qs1, time) - hypoexp(qs2, time) + 1e-8);
  return lik;
}


double loglik_only_seq(const Eigen::MatrixXd& theta, const seq_t& seq, double time) {
  int n = theta.rows();
  VectorXdual tdual(theta.rows()*theta.cols());
  for(auto i=0; i < theta.rows(); i++) {
    for(auto j=0; j < theta.cols(); j++) {
      tdual(j + i*theta.cols()) = theta(i, j);
    }
  }
  auto lik = loglik_seq_aux(tdual, seq, time, n);
  return (double)lik;
}


std::pair<double, Eigen::MatrixXd> loglik_seq(const Eigen::MatrixXd& theta, const seq_t& seq, double time) {
  int n = theta.rows();
  VectorXdual tdual(theta.rows()*theta.cols());
  for(auto i=0; i < theta.rows(); i++) {
    for(auto j=0; j < theta.cols(); j++) {
      tdual(j + i*theta.cols()) = theta(i, j);
    }
  }
  
  dual fval;
  auto grad = gradient(loglik_seq_aux, wrt(tdual), at(tdual, seq, time, n), fval);
  Eigen::MatrixXd matgrad(theta.rows(), theta.cols());
  for(auto i=0; i < theta.rows(); i++) {
    for(auto j=0; j < theta.cols(); j++) {
      matgrad(i, j) = grad(j+n*i);
    }
  }

  return std::make_pair((double) fval, matgrad);
}


std::pair<double, Eigen::MatrixXd> loglik_set_aux(const Eigen::MatrixXd& theta, const seq_t& set, std::vector<seq_t> perms, double time) {
  auto n = theta.rows();
  std::vector<double> liks(perms.size());
  std::vector<Eigen::MatrixXd> grads(perms.size());

  for (auto i=0; i < perms.size(); i++) {
    auto res = loglik_seq(theta, perms[i], time);
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


std::pair<double, Eigen::MatrixXd> loglik_set(const Eigen::MatrixXd& theta, const seq_t& set, double time) {
  auto perms = all_perms(set);
  return loglik_set_aux(theta, set, perms, time);
}


std::pair<double, Eigen::MatrixXd> loglik_data(const Eigen::MatrixXd& theta, const std::vector<seq_t>& data, const std::vector<double>& times) {
  auto n = theta.rows();
  double lik = 0;
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);
  //#pragma omp parallel for reduction(+:lik,grad)
  for (auto i=0; i < data.size(); i++) {
    auto res = loglik_set(theta, data[i], times[i]);
    lik += res.first;
    grad += res.second;
  }
  lik /= data.size();
  grad /= data.size();
  return std::make_pair(lik, grad);
}


#pragma omp declare reduction(+: Eigen::Vector4d: omp_out=omp_out+omp_in) initializer(omp_priv = omp_orig)


std::pair<double, Eigen::Vector4d> loglik(const std::vector<double>& x, const std::vector<seq_t>& data, const std::vector<double>& times) {
  double val = 0;
  Eigen::Vector4d grad = Eigen::Vector4d::Zero(4);

  dual t1 = x[0];
  dual t2 = x[1];
  dual t12 = x[2];
  dual t21 = x[3];

  //#pragma omp parallel for reduction(+:grad,val)
  for (auto i=0; i < data.size(); i++) {
    dual t = times[i];
    if (data[i].size() == 0) {
      auto v = p0(x[0], x[1], x[2], x[3], t);
      val += (double) v;
      Eigen::Vector4d g;
      g << (double) derivative(p0, wrt(t1), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p0, wrt(t2), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p0, wrt(t12), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p0, wrt(t21), at(t1, t2, t12, t21, times[i]));
      grad += g;
    } else if (data[i].size() == 1 && data[i][0] == 0) {
      auto v = p1(x[0], x[1], x[2], x[3], t);
      val += (double) v;
      Eigen::Vector4d g;
      g << (double) derivative(p1, wrt(t1), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p1, wrt(t2), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p1, wrt(t12), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p1, wrt(t21), at(t1, t2, t12, t21, times[i]));
      grad += g;
    } else if (data[i].size() == 1 && data[i][0] == 1) {
      auto v = p2(x[0], x[1], x[2], x[3], t);
      val += (double) v;
      Eigen::Vector4d g;
      g << (double) derivative(p2, wrt(t1), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p2, wrt(t2), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p2, wrt(t12), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p2, wrt(t21), at(t1, t2, t12, t21, times[i]));
      grad += g;
    } else {
      auto v = p12(x[0], x[1], x[2], x[3], t);
      val += (double) v;
      Eigen::Vector4d g;
      g << (double) derivative(p12, wrt(t1), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p12, wrt(t2), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p12, wrt(t12), at(t1, t2, t12, t21, times[i])),
        (double) derivative(p12, wrt(t21), at(t1, t2, t12, t21, times[i]));
      grad += g;
    }
  }

  grad /= -1.0*data.size();
  val /= -1.0*data.size();
  
  return std::make_pair(val, grad);
}


PYBIND11_MODULE(tdiff, m) {
  m.def("loglik", &loglik);
  m.def("loglik_only_seq", &loglik_only_seq);
  m.def("loglik_seq", &loglik_seq);
  m.def("loglik_set", &loglik_set);
  m.def("loglik_data", &loglik_data);
}
