#include <cmath>
#include "autodiff/forward.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <omp.h>
#include "eigen/Eigen/Dense"


using namespace autodiff;


typedef std::vector<int> seq_t;


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
}
