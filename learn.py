import numpy as np
import matplotlib.pylab as plt
import argparse
import simplex_projection
import lbfgs
import datasets
import pickle
import util
import sim
from cpp import diff


def plot_mat(theta, ax, labels, full=False, thresh=0.1):
    n = theta.shape[0]
    mat = np.abs(theta)
    if not full:
        np.fill_diagonal(mat, 0)
    else:
        thresh=0
    idxs = [i for i in range(n)
            if (np.max(mat[i, :]) > thresh or np.max(mat[:, i]) > thresh)]
    if idxs == []:
        plt.pause(0.001)
        return
    idxlab = list(zip(idxs, [labels[i] for i in idxs]))
    idxlab = sorted(idxlab, key=lambda x: x[1])
    idxs, labs = zip(*idxlab)
    #util.plot_matrix(np.exp(theta), ax,
    #                 xlabels=labels, ylabels=labels, permx=list(idxs), permy=list(idxs),
    #                 vmin=0, vmid=1, vmax=5, cmap='PuOr_r', notext=False,
    #                 axes_fontsize=8)
    util.plot_matrix(theta, ax,
                     xlabels=labels, ylabels=labels, permx=list(idxs), permy=list(idxs),
                     vmin=-5, vmid=0, vmax=5, cmap='PuOr_r', notext=False,
                     axes_fontsize=8)
    plt.pause(0.001)


class Optimizer:
    GAMMA = 0.9
    
    def __init__(self, grad):
        self.grad = grad

    def run(self, data, niter, xinit, reg, step=0.1, show=False, only_diag=False):
        if show:
            fig, ax = plt.subplots(1, 1, figsize=(11, 11))
            fig.tight_layout()
            plt.gcf().show()
        
        n = xinit.shape[0]
        mom = np.zeros_like(xinit)
        xsol = np.copy(xinit)
        for it in range(niter):
            util.dot()
            xsol += self.GAMMA*mom
            g = self.grad(xsol, data)

            # XXX
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    if i != j and not (i in [0, 1] and j in [0, 1]):
                        g[i, j] = 0
            g[0, 0] = 0
            #
            
            xsol += step*g
            mom = self.GAMMA*mom + step*g
            # L1-projection
            # TODO: Is this slow?
            xvec = util.mat2vec(xsol)
            if only_diag:
                xvec[n:] = 0
            else:
                xvec[n:] = simplex_projection.euclidean_proj_l1ball(xvec[n:], reg)
            xsol = util.vec2mat(xvec, n)
            if it % 50 == 49:
                util.dot10(str(it+1))
            if show and it % 50 == 0:
                ax.clear()
                plot_mat(xsol, ax, labels=data.labels, full=True)

        #lik = diff.loglik_data_full(xsol, data)
        #print('log-lik =', lik[0])
        #
        if show:
            plt.show()
        return xsol


def learn(data, **kwargs):
    niter = kwargs.get('niter', 1000)
    reg = kwargs.get('reg', 26)
    exact = kwargs.get('exact', False)
    nsamples = kwargs.get('nsamples', 50)
    show = kwargs.get('show', False)
    init_theta = kwargs.get('init_theta', None)

    if exact:
        opt = Optimizer(lambda t, x: diff.loglik_data_full(t, x)[1])
    else:
        opt = Optimizer(lambda t, x: diff.loglik_data(t, x, nsamples))

    if init_theta is None:
        init_theta = np.random.uniform(-5, 5, (data.nitems, data.nitems))
        # XXX
        for i in range(init_theta.shape[0]):
            for j in range(init_theta.shape[1]):
                if i != j and not (i in [0, 1] and j in [0, 1]):
                    init_theta[i, j] = 0
        init_theta[0, 0] = 0
        #
    elif init_theta == 'diag':
        th = np.zeros((data.nitems, data.nitems))
        init_theta = opt.run(data, niter=100, reg=10000, xinit=th, only_diag=True)
    theta = opt.run(data, niter=niter, reg=reg, xinit=init_theta, show=show)
    return theta


def get_data():
    data = datasets.hazard()
    #labels = ['TP53(M)', 'MDM2(A)', 'MDM4(A)', 'CDKN2A(D)', 'CDK4(A)',
    #          'NF1(M)', 'IDH1(M)', 'PTEN(M)', 'PTEN(D)', 'EGFR(M)',
    #          'RB1(D)', 'PDGFRA(A)', 'FAF1(D)', 'SPTA1(M)', 'PIK3CA(M)',
    #          'OBSCN(M)', 'CNTNAP2(M)', 'PAOX(M)', 'TP53(D)', 'LRP2(M)']
    #data = data.subset(data.idx(labels))

    if True:
        data = datasets.tcga('gbm', alt=False, mutonly=False)
        if True:
            cutoff = 0.03
            keep = []
            for idx in range(data.nitems):
                if data.marginals[idx] > cutoff:
                    keep.append(idx)
            print(f'genes: {[data.labels[x] for x in keep]}')
        else:
            labels = ['KRAS', 'APC']
            keep = data.idx(labels)
        data = data.subset(keep)
    return data


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    data = get_data()
    print(f'Running on {data}')
    learn(data, show=args.show)


def run_multi():
    nreps = 25
    HAZARD = False
    if HAZARD:
        data = datasets.hazard()
    else:
        origlabels = ['TP53', 'MDM2(A)', 'MDM4(A)', 'CDKN2A/B(D)', 'CDK4(A)',
                      'NF1', 'IDH1', 'PTEN', 'PTEN(D)', 'EGFR',
                      'RB1(D)', 'PDGFRA(A)', 'SPTA1', 'PIK3CA',
                      'OBSCN', 'CNTNAP2', 'PAOX', 'TP53(D)', 'LRP2',
                      'EGFR(A)', 'RB1', 'ATRX']
        data = datasets.tcga('gbmmed', alt=False, mutonly=False)
        origidx = data.idx(origlabels)
        cutoff = 0.03
        keep = []
        for idx in range(data.nitems):
            if data.marginals[idx] > cutoff:
                keep.append(idx)
        #keep = list(set(keep) | set(origidx))
        keep = origidx
        #
        #keep = data.idx(['EGFR', 'EGFR(A)'])
        #
        data = data.subset(keep)
        print(data)
        print(data.labels)
    thetas = []
    for rep in range(nreps):
        print(f'rep {rep}')
        theta = learn(data, niter=1500, reg=30)
        #lik = diff.loglik_data_full(theta, data)
        #print('log-lik =', lik[0])
        thetas.append(theta)
        #
        plt.gcf().set_size_inches(11, 11)
        plot_mat(theta, plt.gca(), labels=data.labels, full=True)
        plt.savefig(f'res_{rep}.png', dpi=300)
        plt.close()
        #
    thetas = np.asarray(thetas)
    tmin = np.min(np.exp(thetas), axis=0)
    tmax = np.max(np.exp(thetas), axis=0)
    tdif = tmax-tmin
    labels = data.labels

    idxlab = list(zip(range(data.nitems), data.labels))
    idxlab = sorted(idxlab, key=lambda x: x[1])
    idxs, labs = zip(*idxlab)
    util.plot_matrix(tdif, plt.gca(),
                     xlabels=labels, ylabels=labels, permx=list(idxs), permy=list(idxs),
                     vmin=0, vmax=1, cmap='Purples', notext=False,
                     axes_fontsize=8)
    plt.show()


def stats(theta, nsamples):
    import base
    lst = base.get_samples(theta, nsamples)

    #seq = base.get_samples(theta, nsamples, seq=True)
    #tot01 = len([x for x in seq if 0 in x[-1] and 1 in x[-1]])
    #s01 = len([x for x in seq if 0 in x[-1] and 1 in x[-1] and x[-1].index(0) < x[-1].index(1)]) / tot01
    #s10 = len([x for x in seq if 0 in x[-1] and 1 in x[-1] and x[-1].index(1) < x[-1].index(0)]) / tot01
    #print(f'0 -> 1: {s01}')
    #print(f'1 -> 0: {s10}')

    me = len([x for x in lst if x == []]) / nsamples
    p0 = len([x for x in lst if x == [0]]) / nsamples
    p1 = len([x for x in lst if x == [1]]) / nsamples
    p2 = len([x for x in lst if x == [2]]) / nsamples
    m0 = len([x for x in lst if 0 in x]) / nsamples
    m1 = len([x for x in lst if 1 in x]) / nsamples
    m2 = len([x for x in lst if 2 in x]) / nsamples
    m01 = len([x for x in lst if 0 in x and 1 in x]) / nsamples
    p01 = len([x for x in lst if set(x) == set([0, 1])]) / nsamples
    p02 = len([x for x in lst if set(x) == set([0, 2])]) / nsamples
    p12 = len([x for x in lst if set(x) == set([1, 2])]) / nsamples
    print(f'm([]) = {me}')
    print(f'm(0) = {m0}')
    print(f'm(1) = {m1}')
    print(f'm(01) = {m01}')
    print(f'm(r) = {m2}')
    print(f'p(0) = {p0}')
    print(f'p(2) = {p2}')
    print(f'p(01) = {p01}')

    l1 = p0 / me
    l2 = p2 / me
    l3 = p01 / p0

    t22 = p2 / (me * (me + p2))
    a = l3 * (1 + t22)
    t00 = (t22 / l2) - 1
    print(f't22 = {np.log(t22)}')
    print(f'a   = {np.log(a)}')
    print(f't00 = {np.log(t00)}')

    #t00 = t22*p0*(p0+p02)/(me-p0*p2)
    #t00 = p0*t22*(1-me*t22)*(p02+p0) / (me*(p02-me*p02*t22-me*p0*t22))
    #t11 = p1*t22*(1-me*t22)*(p12+p1) / (me*(p12-me*p12*t22-me*p1*t22))
    #print(t11)
    #print(f't00(full) = {np.log(t00)}')
    #print(f't11(full) = {np.log(t11)}')

    r0 = [len(set(x) - set([0, 1])) for x in lst if 0 in x]
    r1 = [len(set(x) - set([0, 1])) for x in lst if 1 in x]

    plt.hist([r0, r1], bins=50, density=True)
    plt.show()


def recover():
    if True:
        theta = np.array([
            [0,   10],
            [0,   -10]])
    else:
        theta = np.array([
            [-1.6,   0],
            [10,   -1.9]])

    nrest = 1
    trest = -1*np.eye(nrest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])
    print(theta)
    stats(theta, 10000)

    if False:
        import base
        import pickle
        #lst = base.get_samples(theta, 2000)
        xs, ps = get_dist(theta)
        #xs = list(util.powerset(range(2+nrest)))
        #ps = base.pt(theta)
        lst = np.random.choice(xs, 1000, replace=True, p=ps)
        with open('data.pcl', 'wb') as fout:
            pickle.dump(lst, fout)
    else:
        import pickle
        with open('data_contour.pcl', 'rb') as fin:
            lst = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=2+nrest)

    lik = diff.loglik_data_full(theta, data)[0]
    print(f'Log-lik (true) = {lik}')
    
    th = learn(data, show=True, niter=20000, reg=3.1, exact=True)
    print(th)
    with open('th.pcl', 'wb') as fout:
        pickle.dump(th, fout)
    lik = diff.loglik_data_full(th, data)[0]
    print(f'===> Log-lik (est) = {lik}')
    stats(th, 10000)


def check_liks():
    t1 = np.array([
        [-1, 10,  0],
        [0,  -10, 0],
        [0,  0,  -1]
    ])
    t2 = np.array([
        [-1.428, 0,  0],
        [4.977,  -1.8128, 0],
        [0,  0,  -1.07]
    ])

    import base
    lst = base.get_samples(t1, 100000)
    data = datasets.Data.from_list([lst], nitems=3)
    lik1 = diff.loglik_data_full(t1, data)[0]
    lik2 = diff.loglik_data_full(t2, data)[0]
    print(f'L1 = {lik1}, L2 = {lik2}')


def check_extra(data, genes):
    idxs = data.idx(genes)
    r0 = [len(set(x) - set(idxs)) for x in data if idxs[0] in x]
    r1 = [len(set(x) - set(idxs)) for x in data if idxs[1] in x]

    plt.hist([r0, r1], bins=100, density=True, label=genes)
    plt.legend()
    plt.show()


def get_dist(theta):
    n = theta.shape[0]
    ps = []
    xs = list(util.powerset(range(n)))
    for s in xs:
        ps.append(diff.loglik_set_full(theta, list(s))[0])
    import scipy.special
    logz = scipy.special.logsumexp(ps)
    ps = np.asarray(ps)
    ps = np.exp(ps - logz)
    return xs, ps


def plot_dist(xs, ps):
    gap = 2 / len(xs)
    for i, p in enumerate(ps):
        x = np.arange(len(xs))
        plt.bar(x + i*gap, p, width=gap)
    plt.xticks(ticks=list(range(len(xs))), labels=[str(x) for x in xs])
    plt.show()


def dists3():
    t1 = np.array([
        [0, 10, 0],
        [0,  -10, 0],
        [0,  0, -1]
    ])
    t2 = np.array([
        [0.0544, 3, 0],
        [0,  -3.133, 0],
        [0,  0, -0.9487]
    ])
    t3 = np.array([
        [-0.272, 0, 0],
        [3,  -1.06, 0],
        [0,  0, -0.945]
    ])
    xs, p1 = get_dist(t1)
    print('p1 =', p1)
    xs, p2 = get_dist(t2)
    xs, p3 = get_dist(t3)

    d1 = 0.5*np.sum(np.abs(p1-p2))
    d2 = 0.5*np.sum(np.abs(p1-p3))
    print(f'TV1 = {d1} | TV2 = {d2}')
    plot_dist(xs, [p1, p2, p3])    


def dists(nrest):
    t1 = np.array([
        [0,   10],
        [0,   -10]])
    trest = -1*np.eye(nrest)
    t1 = np.block([[t1, np.zeros((2, nrest))],
                   [np.zeros((nrest, 2)), trest]])
    with open('t2.pcl', 'rb') as fin:
        t2 = pickle.load(fin)
    with open('t3.pcl', 'rb') as fin:
        t3 = pickle.load(fin)

    with open('data.pcl', 'rb') as fin:
        lst = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=2+nrest)

    lik = diff.loglik_data_full(t1, data)[0]
    print(f'Log-lik (1) = {lik}')
    lik = diff.loglik_data_full(t2, data)[0]
    print(f'Log-lik (2) = {lik}')
    lik = diff.loglik_data_full(t3, data)[0]
    print(f'Log-lik (3) = {lik}')
    
    xs, p1 = get_dist(t1)
    xs, p2 = get_dist(t2)
    xs, p3 = get_dist(t3)

    d1 = 0.5*np.sum(np.abs(p1-p2))
    d2 = 0.5*np.sum(np.abs(p1-p3))
    print(f'TV1 = {d1} | TV2 = {d2}')
    #plot_dist(xs, [p1, p2, p3])


def flik_2(x, data, nrest):
    theta = np.array([
        [0, x[0]],
        [x[1], -x[0]]
    ])
    trest = -1*np.eye(nrest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])

    lik = diff.loglik_data_full(theta, data)
    lik -= 0.03 * (np.abs(x[0]) + np.abs(x[1]))
    return -lik[0]


def flik_3_new(x, data, reg):
    nrest = data.nitems-2
    theta = np.array([
        [0, x[1]],
        [x[2], x[0]]
    ])
    trest = -1*np.eye(nrest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])

    lik = diff.loglik_data_full(theta, data)[0]
    lik -= reg * (np.abs(x[1]) + np.abs(x[2]))
    return lik


def get_lik_new(data, ngrid, reg):
    lo, hi = -10, 10
    t0, t1, t2 = np.meshgrid(np.linspace(-5, -1, ngrid),
                             np.linspace(0, 4, ngrid),
                             np.linspace(-1, 2.5, ngrid),
                             indexing='ij')
    lik = np.zeros_like(t1)
    for i in range(ngrid):
        print(i)
        for j in range(ngrid):
            for k in range(ngrid):
                ts = [t0[i, j, k], t1[i, j, k], t2[i, j, k]]
                lik[i, j, k] = flik_3_new(ts, data, reg)

    x12 = t1[0, :, :]
    y12 = t2[0, :, :]
    lik12 = -np.amax(lik, axis=0)

    x02 = t0[:, 0, :]
    y02 = t2[:, 0, :]
    lik02 = -np.amax(lik, axis=1)

    x10 = t1[:, :, 0]
    y10 = t0[:, :, 0]
    lik10 = -np.amax(lik, axis=2)
    
    return (x12, y12, lik12), (x02, y02, lik02), (x10, y10, lik10)


def flik_full(x, g, data, times=None):
    nrest = data.nitems-2
    theta = np.array([
        [x[0], x[2]],
        [x[3], x[1]]
    ])
    trest = -1*np.eye(nrest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])

    if times is None:
        lik = diff.loglik_data_full(theta, data)
    else:
        lik = diff.loglik_data_full(theta, data, times)
    if g is not None:
        g[0] = -lik[1][0, 0]
        g[1] = -lik[1][1, 1]
        g[2] = -lik[1][0, 1]
        g[3] = -lik[1][1, 0]
    return -lik[0]


def plot_max():
    nreps = 100
    t1s, t2s = [], []
    for i in range(nreps):
        print(i)
        t1, t2, _ = plot_lik(plot=False)
        t1s.append(t1)
        t2s.append(t2)
    plt.plot(t1s, t2s, 'bo', alpha=0.5)
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.show()


def closed_form(data, *args, **kwargs):
    nsamples = len(data)
    me = len([x for x in data if x == []]) / nsamples
    p0 = len([x for x in data if x == [0]]) / nsamples
    p1 = len([x for x in data if x == [1]]) / nsamples
    p2 = len([x for x in data if x == [2]]) / nsamples
    p01 = len([x for x in data if set(x) == set([0, 1])]) / nsamples
    p02 = len([x for x in data if set(x) == set([0, 2])]) / nsamples
    p12 = len([x for x in data if set(x) == set([1, 2])]) / nsamples

    t22 = p2 / (me * (me + p2))
    t00 = p0*t22*(1-me*t22)*(p02+p0) / (me*(p02-me*p02*t22-me*p0*t22))
    t11 = p1*t22*(1-me*t22)*(p12+p1) / (me*(p12-me*p12*t22-me*p1*t22))
    t01 = (1/t11)*(me*t00/p0 - t22 - 1)
    t10 = (1/t00)*(me*t11/p1 - t22 - 1)

    ts = np.log(np.asarray([t11, t01, t10]))
    return ts, 0


def sample_data(xs, ps, nrest, ndata):
    if True:
        lst = np.random.choice(xs, ndata, replace=True, p=ps)
        with open('data.pcl', 'wb') as fout:
            pickle.dump(lst, fout)
    else:
        with open('data_two_opt.pcl', 'rb') as fin:
            lst = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=2+nrest)
    return data

def save_lik(x, g, fx, *args, **kwargs):
    sav = kwargs.get('sav')
    gsav = kwargs.get('gsav')
    #print('x->', x)
    #print('g->', g)
    sav.append(x.copy())
    gsav.append(g.copy())


def max_flik_full(data, reg, times=None):
    import scipy.optimize
    fun = lambda x, g: flik_full(x, g, data, times)
    x0 = np.random.uniform(-2, 2, 4)
    #x0 = np.array([-2.48745317, 2.41311082, 0.87718485])# + np.random.uniform(-0.5, 0.5, 3)

    sav = [x0.copy()]
    gsav = []
    if reg > 0:
        #sol = lbfgs.fmin_lbfgs(fun, x0=x0, epsilon=1e-8,
        #                       max_linesearch=200,
        #                       min_step=1e-60,
        #                       progress=lambda *args: save_lik(*args, sav=sav))
        #print('UNC DONE ======================')
        sol = x0
        for i in range(4):
            sol = lbfgs.fmin_lbfgs(fun, x0=sol,
                                   max_linesearch=100,
                                   min_step=1e-40,
                                   #ftol=1e-6,
                                   #gtol=1e-4,
                                   #past=20,
                                   #delta=1e-20,
                                   orthantwise_c=reg,
                                   orthantwise_start=2,
                                   progress=lambda *args: save_lik(*args, sav=sav, gsav=gsav))
        val = -flik_full(sol, None, data, times) - reg*(np.abs(sol[2]) + np.abs(sol[3]))
    else:
        sol = lbfgs.fmin_lbfgs(fun, x0=x0, epsilon=1e-8,
                               progress=lambda *args: save_lik(*args, sav=sav))
        val = -flik_full(sol, None, data, times)
    return sol, val, (sav, gsav)


def get_theta(nrest):
    theta = np.array([
        [0,   5],
        [0,   -5]
    ])
    trest = 0*np.eye(nrest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])
    return theta


def save_data(nreps, ndata, nrest):
    theta = get_theta(nrest)
    for i in range(nreps):
        lst, times = sim.draw(theta, ndata, time=True)
        with open(f'synth/data_{i}.pcl', 'wb') as fout:
            pickle.dump((lst, times, 2+nrest), fout)


def get_data(i):
    with open(f'synth/data_{i}.pcl', 'rb') as fin:
        lst, times, nitems = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=nitems)
    return data, times


def plot_max2():
    reg = 0.01

    nreps = 200
    ndata = 2000
    nrest = 7
    
    res = []
    vals = []
    savs = []
    gsavs = []

    if False:
        import os, shutil
        shutil.rmtree('synth', ignore_errors=True)
        os.mkdir('synth')
        save_data(nreps, ndata, nrest)
    
    for i in range(nreps):
        data, times = get_data(i)
        data = data.subset([0, 1, 2, 3])
        #import base
        #lst = base.get_samples(theta, ndata)

        r, val, (sav, gsav) = max_flik_full(data, reg, times)
        #r, val = closed_form(data)

        print(f'{i} | val = {val}')
        res.append(r)
        vals.append(val)
        savs.append(sav)
        gsavs.append(gsav)

    maxval = np.max(vals)
    minval = np.min(vals)
    cs = [min(0.1, maxval-v) for v in vals]
    
    t0 = [r[0] for r in res]
    t1 = [r[1] for r in res]
    t2 = [r[2] for r in res]
    t3 = [r[3] for r in res]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.tight_layout()

    if False:
        res12, res02, res10 = get_lik_new(data, ngrid=30, reg=reg)

        lik = res12[2]
        levels = np.linspace(-np.min(lik)-0.05, -np.min(lik), 60)

        ax[0, 0].contourf(res12[0], res12[1], -res12[2], levels=levels, cmap='bone')
        ax[0, 1].contourf(res02[0], res02[1], -res02[2], levels=levels, cmap='bone')
        ax[1, 0].contourf(res10[0], res10[1], -res10[2], levels=levels, cmap='bone')

    if False:
        for i, sav in enumerate(savs):
            if sav[-1][1] < 2.5:
                print('GRAD:', gsavs[i][-1])
                print('SOL:', sav)
                for i in range(len(sav)-1):
                    ax[0, 0].plot([sav[i][1], sav[i+1][1]], [sav[i][2], sav[i+1][2]], 'b.-', alpha=0.2)
                    ax[0, 1].plot([sav[i][0], sav[i+1][0]], [sav[i][2], sav[i+1][2]], 'b.-', alpha=0.2)
                    ax[1, 0].plot([sav[i][1], sav[i+1][1]], [sav[i][0], sav[i+1][0]], 'b.-', alpha=0.2)

    theta = get_theta(nrest)
    ax[0, 0].scatter(t2, t3, c=cs, alpha=0.5)
    ax[0, 0].plot(theta[0, 1], theta[1, 0], 'rx')
    ax[0, 1].scatter(t1, t3, c=cs, alpha=0.5)
    ax[0, 1].plot(theta[1, 1], theta[1, 0], 'rx')
    ax[1, 0].scatter(t2, t1, c=cs, alpha=0.5)
    ax[1, 0].plot(theta[0, 1], theta[1, 1], 'rx')
    ax[1, 1].scatter(t0, t3, c=cs, alpha=0.5)
    ax[1, 1].plot(theta[0, 0], theta[1, 0], 'rx')
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlim((-20, 20))
            ax[i, j].set_ylim((-20, 20))
    print(f'Max dif: {maxval-minval}')
    cnt1 = len([x for x in t3 if -0.1 < x < 0.1])
    cnt2 = len([x for x in t2 if -0.1 < x < 0.1])
    print(f'cnt1 = {cnt1}, cnt2 = {cnt2}')
    plt.show()


if __name__ == '__main__':
    #recover()
    #plot_lik()
    #plot_max()
    plot_max2()

    #data = get_data()
    #print(data)
    #check_extra(data, ['CDK4(A)', 'MDM2(A)'])
    #check_liks()
