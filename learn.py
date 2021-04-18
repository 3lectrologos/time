import numpy as np
import matplotlib.pylab as plt
import argparse
import simplex_projection
import lbfgs
import datasets
import pickle
import util
import sim
import joblib
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
    BUF_SIZE = 200
    MIN_DIF = 0.03

    def init_run(self, xinit, show):
        if show:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 11))
            self.fig.tight_layout()
            plt.gcf().show()
        self.thetas = [np.zeros_like(xinit) for i in range(self.BUF_SIZE)]

    def check_term(self, it, xsol):
        self.thetas[it % self.BUF_SIZE] = xsol
        maxdif = np.abs(xsol - self.thetas[(it+1) % self.BUF_SIZE]).max()
        #
        #arg = np.abs(xsol - thetas[(it+1) % self.BUF_SIZE]).argmax()
        #
        #print(maxdif, '--', arg)
        if maxdif < self.MIN_DIF:
            return True
        else:
            return False

    def plot(self, it, show, data, xsol):
        if it % 50 == 49:
            util.dot10(str(it+1))
        if show and it % 100 == 0:
            self.ax.clear()
            plot_mat(xsol, self.ax, labels=data.labels, full=True)


class NAGOptimizer(Optimizer):
    GAMMA = 0.9
    
    def __init__(self, grad):
        self.grad = grad

    def run(self, data, niter, xinit, step, reg, show=False, only_diag=False, verbose=True):
        self.init_run(xinit, show)
        n = xinit.shape[0]
        mom = np.zeros_like(xinit)
        xsol = np.copy(xinit)
        for it in range(niter):
            util.dot()
            xsol += self.GAMMA*mom
            g = self.grad(xsol, data)
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


class AdaOptimizer(Optimizer):
    EPS = 1e-6

    def __init__(self, grad):
        self.grad = grad

    def run(self, data, niter, xinit, step, reg, show=False, only_diag=False, verbose=True):
        self.init_run(xinit, show)
        n = xinit.shape[0]
        xsol = np.copy(xinit)
        hg = np.zeros_like(xsol)
        for it in range(niter):
            if verbose:
                util.dot()
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
    niter = kwargs.get('niter', 1000)
    step = kwargs.get('step', 0.1)
    reg = kwargs.get('reg', 26)
    exact = kwargs.get('exact', False)
    nsamples = kwargs.get('nsamples', 50)
    show = kwargs.get('show', False)
    init_theta = kwargs.get('init_theta', None)

    if exact:
        opt = AdaOptimizer(lambda t, x: diff.loglik_data_full(t, x)[1])
    else:
        opt = AdaOptimizer(lambda t, x: diff.loglik_data(t, x, nsamples))

    if init_theta is None:
        init_theta = np.random.uniform(-2, 2, (data.nitems, data.nitems))
    elif init_theta == 'diag':
        th = np.zeros((data.nitems, data.nitems))
        init_theta = opt.run(data, niter=100, step=step, reg=0, xinit=th, only_diag=True, show=show)
    theta = opt.run(data, niter=niter, step=step, reg=reg, xinit=init_theta, show=show)
    return theta


def recover_one(args, size, it):
    print(size, '--', it)
    data, _, truetheta = get_data(it)
    truetheta = truetheta[np.ix_([0, 1], [0, 1])]
    truetheta = truetheta / np.sum(np.abs(truetheta))
    print(truetheta)
    data = data.subset(list(range(2+size)))
    theta = learn(data, show=args.show, niter=15000, step=1.0, reg=0.01, exact=False, nsamples=50)#, init_theta='diag')
    #plt.gca().clear()
    #plot_mat(theta, plt.gca(), labels=data.labels, full=True)
    #plt.savefig(f'figs/fig_{size}_{it}.png')
    theta = theta[np.ix_([0, 1], [0, 1])]
    theta = theta / np.sum(np.abs(theta))
    dif = np.sum(np.abs(theta-truetheta))
    print('==>', size, '--', it, '-- dif =', dif)
    return dif


def recover_multi(args):
    sizes = [25]
    niters = 20

    #iters = [9, 17, 20, 23, 30, 34, 38, 42, 47, 50, 51, 54, 67, 68, 76]
    iters = range(niters)
    difs = joblib.Parallel(n_jobs=20)(joblib.delayed(recover_one)(args, size, it)
                                      for size in sizes
                                      for it in iters)
    difs = np.asarray(difs)
    print(difs.shape)
    difs = difs.reshape((len(sizes), -1))
    print(difs)
    with open('difs.pcl', 'wb') as fout:
        pickle.dump((sizes, difs), fout)
    meandifs = np.mean(difs, axis=1)
    plt.gca().clear()
    plt.plot(sizes, meandifs, '-o')
    plt.show()


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


def flik_3_new(x, data, times, reg):
    nrest = data.nitems-2
    theta = np.array([
        [0, x[1]],
        [x[2], x[0]]
    ])
    trest = -1*np.eye(nrest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])

    #lik = diff.loglik_data_full(theta, data)[0]
    lik, _ = tdiff.loglik(x, data, times)
    lik = -lik - reg * (np.abs(x[1]) + np.abs(x[2]))
    return lik


def get_lik_new(data, times, ngrid, reg):
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
                lik[i, j, k] = flik_3_new(ts, data, times, reg)

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
    #print('f(x) ->', fx)
    #print('g ->', g)
    sav.append(x.copy())
    gsav.append(g.copy())


from cpp import tdiff
def tdiff_wrapper(x, g, data, times):
    val, grad = tdiff.loglik(x, data, times)
    for i in range(4):
        g[i] = grad[i]
    return val


def max_flik_full(data, reg, times=None):
    import scipy.optimize
    #fun = lambda x, g: flik_full(x, g, data, times)
    #import tmp
    #fun = tmp.likfun(data, times)
    fun = lambda x, g: tdiff_wrapper(x, g, data, times)

    x0 = np.random.uniform(-2, 2, 4)
    #x0 = np.array([0, -3, 3, 0]) + np.random.uniform(-1, 1, 4)

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
                               progress=lambda *args: save_lik(*args, sav=sav, gsav=gsav))
        val = -flik_full(sol, None, data, times)
    return sol, val, (sav, gsav)


def get_theta(nrest):
    theta = np.array([
        [0,   3],
        [0,   -3]
    ])
    tind = np.random.uniform(-3, 0, nrest)
    #trest = -2*np.eye(nrest)
    trest = np.diag(tind)
    print(trest)
    theta = np.block([[theta, np.zeros((2, nrest))],
                      [np.zeros((nrest, 2)), trest]])
    return theta


def save_data(nreps, ndata, nrest):
    theta = get_theta(nrest)
    for i in range(nreps):
        lst, times = sim.draw(theta, ndata, time=True)
        with open(f'synth/data_{i}.pcl', 'wb') as fout:
            pickle.dump((lst, times, 2+nrest, theta), fout)


def get_data(i):
    with open(f'synth/data_{i}.pcl', 'rb') as fin:
        lst, times, nitems, true = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=nitems)
    return data, times, true


def plot_max2():
    reg = 0.01

    nreps = 100
    ndata = 500
    nrest = 100
    
    res = []
    vals = []
    savs = []
    gsavs = []
    difs = []

    if False:
        import os, shutil
        shutil.rmtree('synth', ignore_errors=True)
        os.mkdir('synth')
        save_data(nreps, ndata, nrest)

    for i in range(20):
        data, times, truetheta = get_data(i)
        truetheta = truetheta[np.ix_([0, 1], [0, 1])]
        truetheta = truetheta / np.sum(np.abs(truetheta))
        data = data.subset([0, 1])

        #import base
        #lst = base.get_samples(theta, ndata)

        r, val, (sav, gsav) = max_flik_full(data, reg, times)
        #r, val = closed_form(data)

        print(f'{i} | val = {val}')
        res.append(r)
        theta = np.array([
            [r[0], r[2]],
            [r[3], r[1]]
        ])
        theta = theta / np.sum(np.abs(theta))
        dif = np.sum(np.abs(theta-truetheta))
        difs.append(dif)
        vals.append(val)
        savs.append(sav)
        gsavs.append(gsav)

    print('meandif =', np.mean(difs))

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
        res12, res02, res10 = get_lik_new(data, times, ngrid=10, reg=reg)

        lik = res12[2]
        levels = np.linspace(-np.min(lik)-2, -np.min(lik), 60)

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
    ax[1, 1].scatter(t0, t1, c=cs, alpha=0.5)
    ax[1, 1].plot(theta[0, 0], theta[1, 1], 'rx')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--rec', action='store_true')
    parser.add_argument('--recall', action='store_true')
    args = parser.parse_args()
    if args.rec:
        #iters = [9, 17, 20, 23, 30, 34, 38, 42, 47, 50, 51, 54, 67, 68, 76]
        recover_one(args, 60, 7)
    elif args.recall:
        recover_multi(args)
    elif args.plot:
        plot_max2()

