import numpy as np
import matplotlib.pylab as plt
import itertools
import collections
import pandas as pd
import lbfgs
import sim
import util
import argparse
from datasets import Data


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s)+1))


def mat2vec(mat):
    uidx = np.triu_indices(mat.shape[0], k=1)
    lidx = np.tril_indices(mat.shape[0], k=-1)
    upper = mat[uidx]
    lower = mat[lidx]
    diag = np.diag(mat)
    return np.hstack((diag, upper, lower))


def vec2mat(vec, n):
    uidx = np.triu_indices(n, k=1)
    lidx = np.tril_indices(n, k=-1)
    didx = np.diag_indices(n)
    ntri = uidx[0].shape[0]
    diag = vec[:n]
    upper = vec[n:n+ntri]
    lower = vec[n+ntri:]
    mat = np.zeros((n, n))
    mat[uidx] = upper
    mat[lidx] = lower
    mat[didx] = diag
    return mat


def get_pdata(data):
    n = data.nitems
    freqs = dict(zip(powerset(range(n)), [0]*(2**n)))
    for d in data:
        freqs[tuple(d)] += 1.0
    freqs = np.array([freqs[key] for key in powerset(range(n))])
    freqs /= len(data)
    return freqs


def idx_to_set(n):
    sets = powerset(range(n))
    return dict(zip(range(2**n), sets))


def loglik(xvec, g, pdata):
    n = int(np.sqrt(xvec.shape[0]))
    x = vec2mat(xvec, n)
    Qt = getq(x)
    # NOTE: Careful with the transposition here.
    Rt = np.eye(2**n) - Qt.T
    p0 = np.zeros(2**n)
    p0[0] = 1.0
    pt = np.linalg.solve(Rt, p0)
    res = np.dot(pdata, np.log(pt))
    # Gradient
    if g is None:
        return -res
    rleft = np.linalg.solve(Rt.T, pdata/pt).T
    i2s = idx_to_set(n)
    grad = np.zeros((n, n))
    for w in range(n):
        for v in range(n):
            gradQ = np.zeros((2**n, 2**n))
            for i in range(gradQ.shape[0]):
                for j in range(gradQ.shape[1]):
                    si = set(i2s[i])
                    sj = set(i2s[j])
                    dif = si ^ sj
                    if sj > si and len(dif) == 1:
                        dif = list(dif)[0]
                        if v == dif and ((w == v) or (w in si)):
                            ts = [x[dif, dif]] + [x[p, dif] for p in si]
                            gradQ[i, j] = np.exp(np.sum(ts))
                            gradQ[i, i] -= np.exp(np.sum(ts))
            grad[w, v] = -rleft @ gradQ.T @ pt
    grad = mat2vec(grad)
    for i in range(g.shape[0]):
        g[i] = grad[i]
    return -res


def getq(x):
    n = x.shape[0]
    i2s = idx_to_set(n)
    q = np.zeros((2**n, 2**n))
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            dif = set(i2s[j]) ^ set(i2s[i])
            if set(i2s[j]) > set(i2s[i]) and len(dif) == 1:
                dif = list(dif)[0]
                ts = [x[dif, dif]] + [x[p, dif] for p in i2s[i]]
                q[i][j] = np.exp(np.sum(ts))
        q[i][i] = -np.sum(q[i, :])
    return q


def get_samples(x, nsamples, seq=False):
    n = x.shape[0]
    Q = getq(x)
    samples = [sim.draw(Q) for i in range(nsamples)]
    i2s = idx_to_set(n)
    if not seq:
        samples = [list(i2s[s[-1]]) for s in samples]
    else:
        samples = [[i2s[i] for i in s] for s in samples]
    return samples


def sample_stat(x, nsamples, seq=False):
    samples = get_samples(x, nsamples, seq)
    samples = [tuple(s) for s in samples]
    counter = collections.Counter(samples)
    for s in counter:
        counter[s] /= nsamples
    for key in sorted(counter.keys()):
        print(key, ':', counter[key])


def example_two_genes(x0=None):
    if x0 is None:
        x0 = np.array([[1, 1], [1, 1]])
    data = [[]]*50 + [[0]]*40 + [[1]]*2 + [[0, 1]]*8
    data = Data.from_list([data], nitems=2)
    pdata = get_pdata(data)
    fopt = lambda x, g: loglik(x, g, pdata)
    sol = lbfgs.fmin_lbfgs(fopt, x0=x0, epsilon=1e-10,
                           orthantwise_c=0.005, line_search='wolfe')
    lik = loglik(sol, None, pdata)
    print('loglik =', lik)
    return sol


def two_genes_reps():
    nreps = 10
    for i in range(nreps):
        x0 = np.random.uniform(-5, 5, (2, 2))
        sol = example_two_genes(x0)
        print('exp(sol) =')
        print(np.exp(sol))
        print('-'*50)


def read_data():
    df = pd.read_csv('gbm.csv')
    mat = df.to_numpy().T
    labels = [col for col in df.columns]
    return Data([mat], labels=labels)


def Lambda_struct():
    lst = [[]]*1 + [[0]]*2 + [[0, 1]]*4 + [[0, 2]]*3
    data = Data.from_list([lst], nitems=3)
    return data


def V_struct():
    lst = [[]]*20 + [[0]]*10 + [[1]]*10 + [[0, 1]]*5 + [[0, 2]]*15 + [[1, 2]]*15 + [[0, 1, 2]]*3
    data = Data.from_list([lst], nitems=3)
    return data


def three():
    foo = np.array([
        [1, 2, 2],
        [1, 1, -10],
        [1, -10, 1]
        ])
    #foo = np.random.uniform(-5, 5, (3, 3))
    print(foo)
    sample_stat(foo, 10000)
    print('-'*50)
    lst = [list(a) for a in get_samples(foo, 100)]
    data = Data.from_list([lst], nitems=3)
    return data    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', action='store_true')
    args = parser.parse_args()
    np.set_printoptions(precision=2, suppress=True)

    #data = read_data()
    #data = data.subset(data.idx(['TP53(M)', 'MDM2(A)', 'CDKN2A(D)', 'CDK4(A)', 'NF1(M)', 'IDH1(M)', 'PTEN(M)']))
    #data = data.subset(data.idx(['PDGFRA(A)', 'CDK4(A)']))
    #data = data.subset(data.idx(['TP53(M)', 'IDH1(M)']))

    data = Lambda_struct()
    #util.plot_perm_group(data, range(2), dlen=len(data),
    #                     niter=10000, axes_fontsize=9, title='', perm=True)
    #plt.show()
    n = data.nitems
    pdata = get_pdata(data)
    fopt = lambda x, g: loglik(x, g, pdata)
    x0 = np.random.uniform(-5, 5, (n, n))
    #x0 = np.ones((n, n))

    if args.reg:
        orthantwise_c = 0.01
        print(f'Running with reg. c = {orthantwise_c}')
        sol = lbfgs.fmin_lbfgs(fopt, x0=mat2vec(x0), epsilon=1e-8,
                               orthantwise_c=orthantwise_c,
                               orthantwise_start=n,
                               line_search='wolfe')
    else:
        sol = lbfgs.fmin_lbfgs(fopt, x0=mat2vec(x0), epsilon=1e-8)

    lik = -loglik(sol, None, pdata)
    print('loglik =', lik)
    sol = vec2mat(sol, n)
    print(sol)
    print('exp(sol) =')
    print(np.exp(sol))
    sample_stat(sol, 10000, seq=False)

    #print('-'*50)

    #foo = np.array([
    #    [1, 2, 2],
    #    [1, 1, -10],
    #    [1, -10, 1]
    #    ])
    #sample_stat(foo, 10000)
