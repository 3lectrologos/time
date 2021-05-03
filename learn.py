import numpy as np
import matplotlib.pylab as plt
import argparse
import simplex_projection
import datasets
import pickle
import util
import sim
import joblib
import random
import collections
from cpp import diff


def plot_mat(theta, ax, labels, full=False, thresh=0.1):
    n = theta.shape[0]
    if n > 50:
        print('Warning: n > 50, plotting only first 50 items')
        n = 50
    mat = np.abs(theta)
    if not full:
        np.fill_diagonal(mat, 0)
    else:
        thresh=0
    idxs = [i for i in range(n)
            if (np.max(mat[i, :]) >= thresh or np.max(mat[:, i]) >= thresh)]
    if idxs == []:
        plt.pause(0.001)
        return
    idxlab = list(zip(idxs, [labels[i] for i in idxs]))
    #idxlab = sorted(idxlab, key=lambda x: x[1])
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
    BUF_SIZE = 200
    MIN_DIF = 0.03

    def __init__(self, grad, verbose=True):
        self.grad = grad
        self.verbose = verbose
    
    def init_run(self, xinit, show):
        if show:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 11))
            self.fig.tight_layout()
            plt.gcf().show()
        self.thetas = [np.zeros_like(xinit) for i in range(self.BUF_SIZE)]

    def check_term(self, it, xsol):
        self.thetas[it % self.BUF_SIZE] = xsol
        maxdif = np.abs(xsol - self.thetas[(it+1) % self.BUF_SIZE]).max()
        #print(maxdif)
        #
        #arg = np.abs(xsol - thetas[(it+1) % self.BUF_SIZE]).argmax()
        #
        #print(maxdif, '--', arg)
        if maxdif < self.MIN_DIF:
            return True
        else:
            return False

    def plot(self, it, show, data, xsol):
        if self.verbose and it % 50 == 49:
            util.dot10(str(it+1))
        if show and it % 100 == 0:
            self.ax.clear()
            plot_mat(xsol, self.ax, labels=data.labels, full=True)


class NAGOptimizer(Optimizer):
    GAMMA = 0.9
    PSTEP = 0.995
    
    def reg_L1(self, xmat, reg):
        n = xmat.shape[0]
        xvec = util.mat2vec(xmat)
        if self.only_diag:
            xvec[n:] = 0
        else:
            xvec[n:] = simplex_projection.euclidean_proj_l1ball(xvec[n:], reg)
        return util.vec2mat(xvec, n)

    def reg_L1L2(self, xmat, reg):
        n = xmat.shape[0]
        if self.only_diag:
            xvec = util.mat2vec(xmat)
            xvec[n:] = 0
            xmat = util.vec2mat(xvec, n)
        else:
            #
            xgroup = np.zeros((n*(n-1)//2, 2))
            k = 0
            for i in range(n):
                for j in range(i):
                    xgroup[k, 0] = xmat[i, j]
                    xgroup[k, 1] = xmat[j, i]
                    k += 1
            #
            
            clipped = np.clip(1 - reg/(1e-8 + np.linalg.norm(xgroup, axis=1)),
                              a_min=0, a_max=None)
            xgroup = np.atleast_2d(clipped).T * xgroup

            #
            k = 0
            for i in range(n):
                for j in range(i):
                    xmat[i, j] = xgroup[k, 0]
                    xmat[j, i] = xgroup[k, 1]
                    k += 1
            #
        return xmat

    def reg_L1Linf(self, xmat, reg):
        n = xmat.shape[0]
        if self.only_diag:
            xvec = util.mat2vec(xmat)
            xvec[n:] = 0
            xmat = util.vec2mat(xvec, n)
        else:
            for i in range(n):
                for j in range(i):
                    v = np.array([xmat[i, j], xmat[j, i]])
                    alpha = simplex_projection.euclidean_proj_l1ball(v, reg)
                    xmat[i, j] -= alpha[0]
                    xmat[j, i] -= alpha[1]
        return xmat

    def reg_Lp(self, xmat, reg, p):
        if self.only_diag:
            n = xmat.shape[0]
            xvec = util.mat2vec(xmat)
            xvec[n:] = 0
            xmat = util.vec2mat(xvec, n)
            return xmat
        else:
            NITER = 50
            u = np.ones_like(xmat)
            for i in range(NITER):
                u =  xmat*u*u / (p*reg*np.power(np.abs(u), p) + u*u + 1e-10)
            idx = u*(u-2*xmat) + 2*reg*np.power(np.abs(u), p)
            u[idx >= 0] = 0
            xdiag = np.diag(np.diag(xmat))
            np.fill_diagonal(u, 0)
            u += xdiag
            return u

    def run(self, data, niter, xinit, step, reg, show=False, only_diag=False):
        preg, lreg = reg
        self.only_diag = only_diag
        self.init_run(xinit, show)
        n = xinit.shape[0]
        mom = np.zeros_like(xinit)
        xsol = np.copy(xinit)
        for it in range(niter):
            if self.verbose:
                util.dot()
            xsol += self.GAMMA*mom
            g = self.grad(xsol, data)
            xsol += step*g
            mom = self.GAMMA*mom + step*g
            # L1-projection
            xsol = self.reg_Lp(xsol, lreg*step, preg)
            #xsol = self.reg_L1(xsol, 3)
            # Check termination
            if it > 1000:
                if self.check_term(it, xsol):
                    break
            if it > 2000:
                step *= self.PSTEP
            # Plot stuff
            self.plot(it, show, data, xsol)

        #lik = diff.loglik_data_full(xsol, data)
        #print('log-lik =', lik[0])
        #
        if show:
            plt.show()
        return xsol


class AdaOptimizer(Optimizer):
    EPS = 1e-6

    def run(self, data, niter, xinit, step, reg, show=False, only_diag=False):
        self.init_run(xinit, show)
        n = xinit.shape[0]
        xsol = np.copy(xinit)
        hg = np.zeros_like(xsol)
        for it in range(niter):
            if self.verbose:
                util.dot()
            # FIXME: Need to separate regularized from non-regularized part
            g = self.grad(xsol, data)
            hg += np.square(g)
            shg = np.sqrt(hg)
            sreg = step*reg / (self.EPS+shg)
            xsol += step*g / (self.EPS+shg)
            xvec = util.mat2vec(xsol)
            rvec = util.mat2vec(sreg)
            if only_diag:
                xvec[n:] = 0
            else:
                xvec[n:] = np.sign(xvec[n:]) * np.clip(np.abs(xvec[n:]) - rvec[n:], 0, None)
            xsol = util.vec2mat(xvec, n)
            # Check termination
            if self.check_term(it, xsol):
                break
            # Plot stuff
            self.plot(it, show, data, xsol)

        #lik = diff.loglik_data_full(xsol, data)
        #print('log-lik =', lik[0])
        #
        if show:
            plt.show()
        return xsol


def learn(data, **kwargs):
    fgrad = kwargs.get('fgrad', None)
    niter = kwargs.get('niter', 1000)
    step = kwargs.get('step', 0.1)
    reg = kwargs.get('reg', 26)
    exact = kwargs.get('exact', False)
    nsamples = kwargs.get('nsamples', 50)
    show = kwargs.get('show', False)
    init_theta = kwargs.get('init_theta', None)
    verbose = kwargs.get('verbose', True)

    if fgrad is not None:
        opt = NAGOptimizer(fgrad, verbose)
    elif exact:
        opt = NAGOptimizer(lambda t, x: diff.loglik_data_full(t, x)[1], verbose)
    else:
        opt = NAGOptimizer(lambda t, x: diff.loglik_data(t, x, nsamples), verbose)

    if init_theta is None:
        init_theta = np.random.uniform(-0.1, 0.1, (data.nitems, data.nitems))
    elif init_theta == 'diag':
        th = np.zeros((data.nitems, data.nitems))
        init_theta = opt.run(data, niter=101, step=1, reg=(1, 0), xinit=th, only_diag=True, show=show)
        th = np.random.uniform(-0.1, 0.1, (data.nitems, data.nitems))
        np.fill_diagonal(th, 0)
        init_theta += th
    theta = opt.run(data, niter=niter, step=step, reg=reg, xinit=init_theta, show=show)
    return theta


def cval(data, reg, nfolds):
    import math
    import random
    batch_size = math.ceil(len(data)/nfolds)
    folds = list(data.batch(batch_size))
    liks = []
    for i in range(nfolds):
        print(f'Fold {i}')
        rest = set(range(nfolds)) - set([i])
        dtrain = datasets.Data.merge([folds[r] for r in rest])
        dtest = folds[i]
        theta = learn(dtrain, show=False, niter=3000, step=1.0, reg=reg, exact=False,
                      nsamples=300, init_theta='diag', verbose=False)
        lik = diff.loglik_data_full(theta, dtest)[0]
        liks.append(lik)
    return liks


def run_cval():
    it = 2
    regs = [0.001, 0.01, 0.02, 0.03, 0.1]
    regparams = [(0.1, r) for r in regs]
    size = 10
    nfolds = 5
    data, _, _, ndep = get_data(it)
    #ind = np.random.choice(list(range(ndep, data.nitems)), size, replace=False)
    ind = list(range(ndep, ndep+size))
    data = data.subset(list(range(ndep)) + list(ind))
    res = joblib.Parallel(n_jobs=len(regs))(joblib.delayed(cval)(data, reg, nfolds)
                                            for reg in regparams)
    res = np.array(res).reshape((-1, nfolds))
    means = np.mean(res, axis=1)
    stds = np.std(res, axis=1)
    plt.errorbar(regs, means, yerr=stds, fmt='o-', capsize=3, linewidth=2)
    plt.gca().set_xscale('log')
    plt.show()


Result = collections.namedtuple('Result', ['data', 'ndep', 'truetheta', 'theta', 'dif'])


def recover_one(args, size, it, rep, feval=None):
    print('Running |', size, '--', it, '-- rep', rep)
    #data, _, truetheta, ndep = get_data(it)
    data = get_real()
    truetheta = None
    ndep = 2
    if True:
        choices = list(range(ndep, data.nitems))
        random.seed(rep)
        random.shuffle(choices)
        ind = choices[:size]
    else:
        ind = [77, 32, 80, 88, 94, 37, 96, 63, 27, 40]
        foo = set(range(30, 50))
        ind = list(set(ind) | foo)
    print('IND ==>', ind)
    data = data.subset(list(range(ndep)) + list(ind))
    theta = learn(data, show=args.show, niter=3000, step=0.5, reg=(0.2, 0.02),
                  exact=False, nsamples=30, init_theta='diag', verbose=False)
    print('Done |', size, '--', it, '-- rep', rep)
    dif = feval(theta, truetheta, ndep)
    return data, ndep, truetheta, theta, dif


def recover_multi(args):
    sizes = [0, 10, 20]#, 1, 2, 3, 4]#, 6, 8, 10]#, 2, 4]#, 6, 8, 10, 15]
    #niters = 19
    nreps = 20

    #iters = range(niters)
    iter = 1
    feval = lambda t1, t2, ndep: t1[0, 1] - t1[1, 0]
    res = joblib.Parallel(n_jobs=20)(joblib.delayed(recover_one)(args, size, iter, rep, feval)
                                     for size in sizes
                                     for rep in range(nreps))
    res = [Result(*r) for r in res]
    with open('difs.pcl', 'wb') as fout:
        pickle.dump((sizes, nreps, res), fout)
    import plot_difs
    plot_difs.plot('difs.pcl')
    plt.show()


def get_real():
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
                if data.marginals[idx] > 0.01:
                    keep.append(idx)
            #print(f'genes: {[data.labels[x] for x in keep]}')
            #keep = data.idx(['MDM2(A)', 'CDK4(A)']) + keep
            #labels = ['TP53', 'MDM2(A)', 'MDM4(A)', 'CDKN2A/B(D)', 'CDK4(A)', 'NF1', 'IDH1', 'PTEN', 'PTEN(D)', 'EGFR', 'RB1(D)', 'PDGFRA(A)', 'PIK3CA', 'TP53(D)']
            labels = ['EGFR', 'EGFR(A)']
            extra = data.idx(labels)
            keep = list(set(keep) - set(extra))
            keep = extra + keep
        else:
            labels = ['TP53', 'IDH1']
            keep = data.idx(labels)
        data = data.subset(keep)
        print(data)
    return data


def run_main(args):
    data = get_real()
    print(f'Running on {data}')
    theta = learn(data, show=args.show, niter=3000, step=0.5, reg=(0.2, 0.01),
                  exact=False, nsamples=100, init_theta='diag', verbose=True)


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


from cpp import tdiff
def tdiff_wrapper(x, g, data, times):
    n = data.nitems
    xmat = util.vec2mat(x, n)
    val, grad = tdiff.loglik_data(xmat, data, times)
    val = -val
    grad = util.mat2vec(grad)
    for i in range(n*n):
        g[i] = -grad[i]
    return val


def get_theta(nrest):
    theta = np.array([
        [0, 2],
        [1, -1]
    ])

    #theta = 3.0*np.array([
    #    [0, 1, 0, 0],
    #    [0, -1, 0, 0],
    #    [0, 0, 0, 1],
    #    [0, 0, 0, -1]
    #])

    #theta = 3.0*np.array([
    #    [0, 1, 1],
    #    [0, -1, 0],
    #    [0, 0, -1],
    #])

    #theta = 3.0*np.array([
    #    [0, 1, 0],
    #    [0, -1, 1],
    #    [0, 0, -1],
    #])

    #theta = np.zeros((5, 5))
    #edges = [(i, j) for i in range(5) for j in range(5) if i != j]
    #idxs = np.random.choice(list(range(len(edges))), 5, replace=False)
    #for i in idxs:
    #    val = np.random.uniform(-3, 3)
    #    theta[edges[i][0], edges[i][1]] = val
    #print(theta)

    ndep = theta.shape[0]
    tind = np.random.uniform(-1, 0, nrest)
    trest = np.diag(tind)
    print(trest)
    theta = np.block([[theta, np.zeros((ndep, nrest))],
                      [np.zeros((nrest, ndep)), trest]])
    return theta, ndep


def save_data(nreps, ndata, nrest):
    theta, ndep = get_theta(nrest)
    for i in range(nreps):
        lst, times = sim.draw(theta, ndata, time=True)
        with open(f'synth/data_{i}.pcl', 'wb') as fout:
            pickle.dump((lst, times, ndep, nrest, theta), fout)


def get_data(i):
    with open(f'synth/data_{i}.pcl', 'rb') as fin:
        lst, times, ndep, nrest, true = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=ndep+nrest)
    return data, times, true, ndep


def plot_max2(show):
    nreps = 20
    ndata = 1000
    nrest = 100
    
    res = []
    difs = []

    if False:
        import os, shutil
        shutil.rmtree('synth', ignore_errors=True)
        os.mkdir('synth')
        save_data(nreps, ndata, nrest)

    for i in [1]:
        data, times, truetheta, ndep = get_data(i)

        #data = data.subset(list(range(50)))
        #foo = [len(d) for d in data]
        #plt.hist(foo, bins=20)
        #plt.show()
        
        truetheta = truetheta[np.ix_(list(range(ndep)), list(range(ndep)))]
        data = data.subset(list(range(ndep)))

        if False:
            t0 = [times[i] for i, d in enumerate(data) if d == [0]]
            t01 = [times[i] for i, d in enumerate(data) if 1 in d and 2 not in d]
            t02 = [times[i] for i, d in enumerate(data) if 2 in d]
            plt.hist(t0, alpha=0.3, bins=50, density=True)
            plt.hist(t01, alpha=0.3, bins=50, density=True)
            plt.hist(t02, alpha=0.3, bins=50, density=True)
            plt.show()

        print(f'Learning {i}')
        fgrad = lambda t, x: tdiff.loglik_data(t, x, times)[1]
        theta = learn(data, fgrad=fgrad, show=show, niter=3000, step=1.0, reg=(0.2, 0.02),
                      exact=True, nsamples=50, init_theta='diag', verbose=False)
        print('theta =')
        print(theta)
        res.append(theta)
        dif = sim.tv_seq(truetheta, theta, ndep, nsamples=10000)
        print('dif =', dif)
        difs.append(dif)

    print('meandif =', np.mean(difs))

    t0 = [r[0, 0] for r in res]
    t1 = [r[1, 1] for r in res]    
    t2 = [r[0, 1] for r in res]
    t3 = [r[1, 0] for r in res]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.tight_layout()

    pairs = [
        # axis,  i1, j1, i2, j2
        ([0, 0], [0, 1], [1, 0]),
        ([0, 1], [0, 0], [1, 1]),
        ([1, 0], [0, 1], [1, 0]),
        ([1, 1], [0, 1], [1, 0]),
        #([0, 1], [0, 2], [2, 0]),
        #([1, 0], [1, 2], [2, 1]),
        #([1, 1], [0, 0], [1, 1])
    ]

    truetheta = get_data(0)[2]
    truetheta = truetheta[np.ix_(list(range(ndep)), list(range(ndep)))]
    for axidx, p1, p2 in pairs:
        t1 = [r[p1[0], p1[1]] for r in res]
        t2 = [r[p2[0], p2[1]] for r in res]
        ax[axidx[0], axidx[1]].scatter(t1, t2, alpha=0.5)
        ax[axidx[0], axidx[1]].plot(truetheta[p1[0], p1[1]], truetheta[p2[0], p2[1]], 'rx')
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlim((-20, 20))
            ax[i, j].set_ylim((-20, 20))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--rec', nargs=3)
    parser.add_argument('--recall', action='store_true')
    parser.add_argument('--cval', action='store_true')
    parser.add_argument('--main', action='store_true')
    args = parser.parse_args()
    if args.rec:
        recover_one(args, int(args.rec[0]), int(args.rec[1]), rep=int(args.rec[2]))
    elif args.recall:
        recover_multi(args)
    elif args.plot:
        plot_max2(args.show)
    elif args.cval:
        run_cval()
    elif args.main:
        run_main(args)
