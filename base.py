import numpy as np
import scipy.linalg
import matplotlib.pylab as plt
import itertools
import collections
import sim
import util
import argparse
import datasets


def get_pdata(data):
    n = data.nitems
    freqs = dict(zip(util.powerset(range(n)), [0]*(2**n)))
    for d in data:
        freqs[tuple(d)] += 1.0
    freqs = np.array([freqs[key] for key in util.powerset(range(n))])
    freqs /= len(data)
    return freqs


def idx_to_set(n):
    sets = util.powerset(range(n))
    return dict(zip(range(2**n), sets))


def pt(x):
    n = x.shape[0]
    Qt = getq(x)
    # NOTE: Careful with the transposition here.
    Rt = np.eye(2**n) - Qt.T
    p0 = np.zeros(2**n)
    p0[0] = 1.0
    pt = np.linalg.solve(Rt, p0)
    return pt


def ptt(x, t):
    n = x.shape[0]
    Qt = getq(x)
    eq = scipy.linalg.expm(t*Qt.T)
    p0 = np.zeros(2**n)
    p0[0] = 1.0
    pt = np.dot(eq, p0)
    return pt


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


def get_samples_old(x, nsamples, seq=False):
    n = x.shape[0]
    Q = getq(x)
    samples = [sim.draw_old(Q) for i in range(nsamples)]
    i2s = idx_to_set(n)
    if not seq:
        samples = [list(i2s[s[-1]]) for s in samples]
    else:
        # TODO: Make into ordered list, instead of list of tuples
        samples = [[i2s[i] for i in s] for s in samples]
    return samples


def get_samples(theta, nsamples, seq=False):
    n = theta.shape[0]
    samples = [sim.draw(theta) for i in range(nsamples)]
    return samples


def sample_stat(x, nsamples, seq=False):
    samples = get_samples(x, nsamples, seq)
    samples = [frozenset(s) for s in samples]
    counter = collections.Counter(samples)
    for s in counter:
        counter[s] /= nsamples
    for key in sorted([sorted(list(key)) for key in counter.keys()]):
        print(key, ':', counter[frozenset(key)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', action='store_true')
    args = parser.parse_args()
    np.set_printoptions(precision=2, suppress=True)

    data = datasets.hazard()
    data = data.subset(data.idx(
        ['TP53(M)', 'MDM2(A)', 'CDKN2A(D)', 'CDK4(A)', 'NF1(M)', 'IDH1(M)', 'PTEN(M)']))
    #data = data.subset(data.idx(['PDGFRA(A)', 'CDK4(A)']))
    #data = data.subset(data.idx(['TP53(M)', 'IDH1(M)']))

    #data = Lambda_struct()
    #util.plot_perm_group(data, range(2), dlen=len(data),
    #                     niter=10000, axes_fontsize=9, title='', perm=True)
    #plt.show()
    n = data.nitems
    pdata = get_pdata(data)
    fopt = lambda x, g: loglik(x, g, pdata)
    np.random.seed(42)
    x0 = np.random.uniform(-5, 5, (n, n))
    #x0 = np.ones((n, n))

    import lbfgs
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
    #sample_stat(sol, 10000, seq=False)

    #print('-'*50)

    #foo = np.array([
    #    [1, 2, 2],
    #    [1, 1, -10],
    #    [1, -10, 1]
    #    ])
    #sample_stat(foo, 10000)
