import numpy as np
import matplotlib.pylab as plt
import itertools
from collections import defaultdict
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

    lse = scipy.special.logsumexp(liks)
    liks = np.asarray(liks)
    liks -= lse
    grd = np.zeros_like(th)
    for w, g in zip(liks, grds):
        grd += np.exp(w)*g
    return lse, grd


def loglik_set_new(th, x):
    ground = set(range(th.shape[0]))
    xset = set(x)
    prev = {frozenset([]): (0, np.zeros_like(th))}
    while True:
        if len(x) == 0:
            break
        next = defaultdict(lambda: ([], []))
        for ps, (pval, pgrad) in prev.items():
            rest = xset - ps
            for r in rest:
                ns = ps | frozenset([r])
                nval = pval
                gval = pgrad.copy()
                sumth = []
                for j in ground - ps:
                    sumth.append(th[j, j] + sum(th[i, j] for i in ps))
                    if j == r:
                        nval += sumth[-1]
                        gval[j, j] += 1
                        for i in ps:
                            gval[i, j] += 1
                lse = scipy.special.logsumexp([0] + sumth)
                nval -= lse
                for k, j in enumerate(ground - ps):
                    gval[j, j] -= np.exp(sumth[k]-lse)
                    for i in ps:
                        gval[i, j] -= np.exp(sumth[k]-lse)
                next[ns][0].append(nval)
                next[ns][1].append(gval)
        for ns, (nval, ngrad) in next.items():
            lse = scipy.special.logsumexp(nval)
            nextval = lse
            nextgrad = np.zeros_like(th)
            for nv, ng in zip(nval, ngrad):
                nextgrad += (np.exp(nv-lse)*ng)
            next[ns] = (nextval, nextgrad)
        prev = next
        if len(prev) == 1:
            break
    assert len(prev) == 1
    assert list(prev.keys())[0] == xset
    sumth = []
    for j in ground - xset:
        sumth.append(th[j, j] + sum(th[i, j] for i in xset))
    lse = scipy.special.logsumexp([0] + sumth)
    lik = list(prev.items())[0][1][0] - lse
    grad = list(prev.items())[0][1][1]
    for r, j in enumerate(ground - xset):
        grad[j, j] -= np.exp(sumth[r]-lse)
        for i in xset:
            grad[i, j] -= np.exp(sumth[r]-lse)
    return lik, grad


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
