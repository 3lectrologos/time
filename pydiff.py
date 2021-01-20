import numpy as np
import matplotlib.pylab as plt
import itertools
import argparse
import scipy.special
import base
import util
from cpp import diff


def loglik_seq(th, x):
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


def loglik_set(th, x):
    res = [loglik_seq(th, xperm)
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


def test():
    n = 7
    np.random.seed(41)
    theta = np.random.uniform(-5, 5, (n, n))
    seq = [2, 4, 5, 3, 0, 1, 6]
    #seq = [2, 0, 1, 5]

    #print('theta =\n', theta)
    #print()

    res, grad = loglik_set(theta, seq)
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
    test()
