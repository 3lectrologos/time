import numpy as np
import matplotlib.pylab as plt
import itertools
import argparse
import scipy.special
import lbfgs
import base
import util
from cpp import diff

import jax.numpy as jnp
import jax.scipy.special as jsp


def q(th, s, a):
    res = th[a, a]
    for i in s:
        res += th[i, a]
    return res


def loglik_seq(th, x):
    n = th.shape[0]
    res = 0
    i = 0
    for i in range(len(x)):
        num = q(th, x[:i], x[i])
        rest = [q(th, x[:i], j)
                for j in set(range(n)) - set(x[:i+1])]
        res += (num - jsp.logsumexp(jnp.array([0, num] + rest)))
    # Final factor
    rest = [q(th, x[:i+1], j)
            for j in set(range(n)) - set(x[:i+1])]
    if len(rest) > 0:
        res -= jsp.logsumexp(jnp.array([0] + rest))
    return res


def loglik_set(th, x):
    lls = [loglik_seq(th, xperm) for xperm in itertools.permutations(x)]
    return jsp.logsumexp(jnp.array(lls))


def loglik(thvec, gout, data, gfun):
    n = int(np.sqrt(thvec.shape[0]))
    th = jnp.array(base.vec2mat(thvec, n))
    val = 0
    grd = np.zeros((n, n))
    for d in data:
        v, g = gfun(th, d)
        val += v
        grd += g
    grd = base.mat2vec(grd)
    if gout is not None:
        for i in range(len(gout)):
            gout[i] = -grd[i]/len(data)
    return -val/len(data)


def loglik_seq_new(th, x):
    ground = set(range(th.shape[0]))
    lik = 0
    grd = np.zeros_like(th)
    for k in range(len(x)+1):
        if k < len(x):
            i = x[k]
            lik += th[i, i]
            grd[i, i] += 1
            for j in x[k+1:]:
                lik += th[i, j]
                grd[i, j] += 1
        sumth = []
        for j in ground - set(x[:k]):
            sumth.append(
                th[j, j] + sum(th[i, j] for i in x[:k]))
        lse = scipy.special.logsumexp([0] + sumth)
        sumth = np.asarray(sumth)
        lik -= lse
        sumth -= lse
        for r, j in enumerate(ground - set(x[:k])):
            grd[j, j] -= np.exp(sumth[r])
            for i in x[:k]:
                grd[i, j] -= np.exp(sumth[r])
    return lik, grd


def loglik_set_new(th, x):
    res = [loglik_seq_new(th, xperm)
           for xperm in itertools.permutations(x)]
    liks, grds = zip(*res)

    print(liks[:10])
                     
    lse = scipy.special.logsumexp(liks)
    liks = np.asarray(liks)
    liks -= lse
    grd = np.zeros_like(th)
    for w, g in zip(liks, grds):
        grd += np.exp(w)*g
    return lse, grd


def loglik_new(thvec, gout, data, nperm=20):
    n = int(np.sqrt(thvec.shape[0]))
    th = base.vec2mat(thvec, n)
    val = 0
    grd = np.zeros((n, n))
    for d in data:
        v, g = diff.loglik_set(th, d, nperm)
        val += v
        grd += g
    grd = base.mat2vec(grd)
    if gout is not None:
        for i in range(len(gout)):
            gout[i] = -grd[i]/len(data)
    return -val/len(data)


def learn():
    #n = 3
    #data = [[]]*1 + [[0]]*2 + [[0, 1]]*4 + [[0, 2]]*3
    
    data = base.read_data()
    data = data.subset(
        data.idx(['TP53(M)', 'MDM2(A)', 'MDM4(A)', 'CDKN2A(D)', 'CDK4(A)',
                  'NF1(M)', 'IDH1(M)', 'PTEN(M)', 'PTEN(D)', 'EGFR(M)',
                  'RB1(D)', 'PDGFRA(A)', 'FAF1(D)', 'SPTA1(M)', 'PIK3CA(M)',
                  'OBSCN(M)', 'CNTNAP2(M)', 'PAOX(M)', 'TP53(D)', 'LRP2(M)']))
    n = data.nitems
    
    fopt = lambda x, g: loglik_new(x, g, data)
    np.random.seed(42)
    x0 = np.random.uniform(-5, 5, (n, n))
    REG = True
    if not REG:
        sol = lbfgs.fmin_lbfgs(fopt, x0=base.mat2vec(x0), epsilon=1e-8)
    else:
        orthantwise_c = 0.01
        sol = lbfgs.fmin_lbfgs(fopt, x0=base.mat2vec(x0), epsilon=1e-8,
                               orthantwise_c=orthantwise_c,
                               orthantwise_start=n,
                               line_search='wolfe')
    lik = -loglik_new(sol, None, data)
    print('loglik =', lik)
    sol = base.vec2mat(sol, n)
    print(sol)
    #print('exp(sol) =')
    #print(np.exp(sol))
    #base.sample_stat(sol, 10000, seq=False)    


def test3():
    n = 7
    np.random.seed(41)
    theta = np.random.uniform(-5, 5, (n, n))
    seq = [2, 4, 5, 3, 0, 1, 6]
    #seq = [2, 0, 1, 5]

    #print('theta =\n', theta)
    #print()

    res, grad = loglik_set_new(theta, seq)
    print('true =', res)
    print('grad =\n', grad)
    print('=======================')
    print()

    from datasets import Data
    data = Data.from_list([[seq]], nitems=n)
    grad = np.zeros(n*n)
    res = base.loglik(base.mat2vec(theta), grad, base.get_pdata(data))
    grad = base.vec2mat(grad, n)
    print('true =', res)
    print('grad =\n', grad)
    print('=======================')
    print()
    
    res, grad = diff.loglik_set(theta, seq, 1000)
    print('approx =', res)
    print('grad =\n', grad)


if __name__ == '__main__':
    test3()
