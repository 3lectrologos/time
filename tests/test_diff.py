import numpy as np
import numpy.testing as nptest
import pydiff
from cpp import diff


def test_diff_seq():
    n = 7
    nreps = 100
    seeds = list(range(nreps))
    for seqlen in range(n):
        for seed in seeds:
            np.random.seed(seed)
            theta = np.random.uniform(-5, 5, (n, n))
            seq = list(np.random.choice(list(range(n)), seqlen, replace=False))
            pylik, pygrad = pydiff.loglik_seq(theta, seq)
            cplik, cpgrad = diff.loglik_seq(theta, seq)
            nptest.assert_almost_equal(pylik, cplik)
            nptest.assert_almost_equal(pygrad, cpgrad)


def test_diff_set():
    n = 5
    nreps = 100
    seeds = list(range(nreps))
    for seqlen in range(n):
        for seed in seeds:
            np.random.seed(seed)
            theta = np.random.uniform(-5, 5, (n, n))
            seq = list(np.random.choice(list(range(n)), seqlen, replace=False))
            pylik, pygrad = pydiff.loglik_set(theta, seq)
            cplik, cpgrad = diff.loglik_set_full(theta, seq)
            nptest.assert_almost_equal(pylik, cplik)
            nptest.assert_almost_equal(pygrad, cpgrad)
