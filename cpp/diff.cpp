#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <omp.h>
#include "eigen/Eigen/Dense"


namespace py = pybind11;


typedef std::set<int> set_t;
typedef std::vector<int> seq_t;
typedef std::vector<double> dseq_t;
typedef std::vector<Eigen::MatrixXd> matseq_t;


const double NOTIME = -1;


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
      dseq_t sumth(rest.size());
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
      dseq_t sumth(rest.size());
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


std::pair<seq_t, double> sample_one(const Eigen::MatrixXd& theta, const seq_t& set, std::mt19937& gen) {
  int n = theta.rows();
  seq_t setrest(set.cbegin(), set.cend());
  std::sort(setrest.begin(), setrest.end());
  seq_t result;
  double logprob = 0;

  for (auto k=0; k < set.size(); k++) {
    dseq_t logprobs(setrest.size());
    int r = 0;
    for (auto i : setrest) {
      for (auto j : setrest) {
        if (i != j) {
          logprobs[r] += theta(i, j);
        }
      }
      r++;
    }

    double logsumprobs = logsumexp(logprobs);
    dseq_t probs(logprobs);
    for (auto i=0; i < probs.size(); i++) {
      probs[i] = exp(probs[i]);
    }
    
    std::discrete_distribution<> dist(probs.cbegin(), probs.cend());
    auto idx = dist(gen);
    auto next = setrest[idx];
    logprob += (logprobs[idx] - logsumprobs);

    // Remove `next` from `rest` and `setrest`, and add to `result`.
    result.push_back(next);
    setrest.erase(setrest.begin()+idx);
  }

  //std::cout << result[0] << " " << result[1] << std::endl;
  return std::make_pair(result, logprob);
}


Eigen::MatrixXd loglik_set(const Eigen::MatrixXd& theta, const seq_t& set, int nperms) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0, 1);
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(theta.rows(), theta.rows());

  int burnin = 0;
  std::vector<seq_t> result;
  //int naccept = 0;
  auto res = sample_one(theta, set, gen);
  auto cur = res.first;
  auto qcur = res.second;
  auto rescur = loglik_seq(theta, cur);
  auto pcur = rescur.first;
  auto gcur = rescur.second;
  
  for(auto i=0; i < burnin+nperms; i++) {
    auto res = sample_one(theta, set, gen);
    auto next = res.first;
    auto qnext = res.second;
    auto resnext = loglik_seq(theta, next);
    auto pnext = resnext.first;
    auto gnext = resnext.second;
    auto paccept = std::min(1.0, exp(pnext-pcur+qcur-qnext));
    if (unif(gen) < paccept) {
      pcur = pnext;
      qcur = qnext;
      gcur = gnext;
      cur = next;
      //if (i >= burnin) naccept++;
    }
    if (i >= burnin) {
      grad += gcur;
    }
  }

  //std::cout << "acc. rate = " << naccept / (1.0*nperms) << std::endl;
  return grad/nperms;
}


Eigen::MatrixXd loglik_set_old(const Eigen::MatrixXd& theta, const seq_t& set, int nperms) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0, 1);
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(theta.rows(), theta.rows());

  int burnin = 0;
  std::vector<seq_t> result;
  //int naccept = 0;
  seq_t cur(set);
  auto rescur = loglik_seq(theta, cur);
  auto pcur = rescur.first;
  auto gcur = rescur.second;
  
  for(auto i=0; i < burnin+nperms; i++) {
    seq_t next(cur);
    std::shuffle(next.begin(), next.end(), gen);
    auto resnext = loglik_seq(theta, next);
    auto pnext = resnext.first;
    auto gnext = resnext.second;
    auto paccept = std::min(1.0, exp(pnext-pcur));
    if (unif(gen) < paccept) {
      pcur = pnext;
      gcur = gnext;
      cur = next;
      //if (i >= burnin) naccept++;
    }
    if (i >= burnin) {
      grad += gcur;
    }
  }

  //std::cout << "acc. rate = " << naccept / (1.0*nperms) << std::endl;
  return grad/nperms;
}


std::pair<double, Eigen::MatrixXd> loglik_set_full_old(const Eigen::MatrixXd& theta, const seq_t& set) {
  auto perms = all_perms(set);
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


std::pair<double, Eigen::MatrixXd> loglik_set_full(const Eigen::MatrixXd& theta, const seq_t& x) {
  auto n = theta.rows();
  set_t ground;
  for (auto i=0; i < n; i++) ground.insert(i);
  set_t xs(x.begin(), x.end());
  std::map<set_t, double> prev;
  std::map<set_t, Eigen::MatrixXd> prevgrad;
  set_t empty;
  prev[empty] = 0;
  prevgrad[empty] = Eigen::MatrixXd::Zero(n, n);

  while (true) {
    if (x.size() == 0) {
      break;
    }
    std::map<set_t, dseq_t> next;
    std::map<set_t, matseq_t> nextgrad;
    for (const auto& [ps, pval] : prev) {
      set_t rest;
      std::set_difference(xs.begin(), xs.end(), ps.begin(), ps.end(),
                          std::inserter(rest, rest.begin()));
      for (auto r : rest) {
        set_t ns(ps);
        ns.insert(r);
        double nval = pval;
        auto gval = prevgrad[ps];
        set_t diff;
        std::set_difference(ground.begin(), ground.end(), ps.begin(), ps.end(),
                            std::inserter(diff, diff.begin()));
        dseq_t sumth(diff.size()+1);
        sumth[diff.size()] = 0;
        int k = 0;
        for (auto j : diff) {
          sumth[k] = theta(j, j);
          for (auto i : ps) {
            sumth[k] += theta(i, j);
          }
          if (j == r) {
            nval += sumth[k];
            gval(j, j) += 1;
            for (auto i : ps) {
              gval(i, j) += 1;
            }
          }
          k++;
        }
        double lse = logsumexp(sumth);
        nval -= lse;
        k = 0;
        for (auto j : diff) {
          gval(j, j) -= exp(sumth[k]-lse);
          for (auto i : ps) {
            gval(i, j) -= exp(sumth[k]-lse);
          }
          k++;
        }
        
        auto found = next.find(ns);
        if (found == next.end()) {
          next[ns] = dseq_t();
          nextgrad[ns] = matseq_t();
        }
        next[ns].push_back(nval);
        nextgrad[ns].push_back(gval);
      }
    }
    prev.clear();
    prevgrad.clear();
    for (const auto& [ns, nval] : next) {
      double lse = logsumexp(nval);
      prev[ns] = lse;
      Eigen::MatrixXd ngrad = Eigen::MatrixXd::Zero(n, n);
      for (auto i=0; i < nval.size(); i++) {
        ngrad += exp(nval[i]-lse)*nextgrad[ns][i];
      }
      prevgrad[ns] = ngrad;
    }
    if (prev.size() == 1) break;
  }

  set_t diff;
  std::set_difference(ground.begin(), ground.end(), xs.begin(), xs.end(),
                      std::inserter(diff, diff.begin()));
  dseq_t sumth(diff.size()+1);
  sumth[diff.size()] = 0;
  int r = 0;
  for (auto j : diff) {
    sumth[r] = theta(j, j);
    for (auto i : xs) {
      sumth[r] += theta(i, j);
    }
    r++;
  }
  double lse = logsumexp(sumth);

  double lik = prev[xs] - lse;
  Eigen::MatrixXd grad = prevgrad[xs];
  r = 0;
  for (auto j : diff) {
    grad(j, j) -= exp(sumth[r]-lse);
    for (auto i : xs) {
      grad(i, j) -= exp(sumth[r]-lse);
    }
    r++;
  }
  return std::make_pair(lik, grad);
}


#pragma omp declare reduction(+: Eigen::MatrixXd: omp_out=omp_out+omp_in) initializer(omp_priv = omp_orig)

std::pair<double, Eigen::MatrixXd> loglik_data_full(const Eigen::MatrixXd& theta, const std::vector<seq_t>& data) {
  auto n = theta.rows();
  double lik = 0;
  Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n, n);
  //#pragma omp parallel for reduction(+:lik,grad)
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
#pragma omp parallel for reduction(+:grad)
  for (auto i=0; i < data.size(); i++) {
    // TODO: Refactor this
    if (data[i].size() < 8) {
      grad += loglik_set_full(theta, data[i]).second;
    } else {
      grad += loglik_set(theta, data[i], nperms);
    }
  }
  grad /= data.size();
  return grad;
}


std::vector<seq_t> draw(const Eigen::MatrixXd& theta, int nsamples) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<seq_t> res;
  auto n = theta.rows();
  for (auto cnt=0; cnt < nsamples; cnt++) {
    seq_t seq;
    double tcur = 0;
    seq_t rest(n);
    std::exponential_distribution<double> dexp(1.0);
    auto tstop = dexp(gen);
    for (auto i=0; i < n; i++) rest[i] = i;
    while (true) {
      if (rest.size() == 0) break;
      dseq_t sumth(rest.size());
      int r = 0;
      for (auto j : rest) {
        sumth[r] = theta(j, j);
        for (auto i : seq) {
          sumth[r] += theta(i, j);
        }
        r++;
      }
      double lse = logsumexp(sumth);
      std::exponential_distribution<double> dist(exp(lse));
      tcur += dist(gen);
      if (tcur > tstop) break;
      dseq_t ps(rest.size());
      for (auto i=0; i < rest.size(); i++) {
        ps[i] = exp(sumth[i]-lse);
      }
      std::discrete_distribution<> dchoice(ps.cbegin(), ps.cend());
      auto idx = dchoice(gen);
      seq.push_back(rest[idx]);
      rest.erase(rest.begin()+idx);
    }
    res.push_back(seq);
  }
  return res;
}


PYBIND11_MODULE(diff, m) {
  m.def("loglik_seq", &loglik_seq, py::return_value_policy::reference_internal);
  m.def("loglik_set_old", &loglik_set_old, py::return_value_policy::reference_internal);
  m.def("loglik_set", &loglik_set, py::return_value_policy::reference_internal);
  m.def("loglik_set_full_old", &loglik_set_full_old, py::return_value_policy::reference_internal);
  m.def("loglik_set_full", &loglik_set_full, py::return_value_policy::reference_internal);
  m.def("loglik_data", &loglik_data, py::return_value_policy::reference_internal);
  m.def("loglik_data_full", &loglik_data_full, py::return_value_policy::reference_internal);
  m.def("draw", &draw, py::return_value_policy::reference_internal);
}
