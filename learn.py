import numpy as np
import matplotlib.pylab as plt
import util
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
        thresh = 0
    idxs = [i for i in range(n)
            if (np.max(mat[i, :]) >= thresh or np.max(mat[:, i]) >= thresh)]
    if idxs == []:
        plt.pause(0.001)
        return
    idxlab = list(zip(idxs, [labels[i] for i in idxs]))
    # idxlab = sorted(idxlab, key=lambda x: x[1])
    idxs, labs = zip(*idxlab)
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
            # self.fig.tight_layout()
            plt.gcf().show()
        self.thetas = [np.zeros_like(xinit) for i in range(self.BUF_SIZE)]

    def check_term(self, it, xsol):
        dif = xsol - self.thetas[it % self.BUF_SIZE]
        maxdif = np.abs(dif).max()
        self.thetas[it % self.BUF_SIZE] = xsol.copy()
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


class AdaOptimizer(Optimizer):
    EPS = 1e-6

    def run(self, data, maxiter, xinit, step, reg, show=False, only_diag=False):
        self.init_run(xinit, show)
        n = xinit.shape[0]
        xsol = np.copy(xinit)
        hg = np.zeros_like(xsol)
        for it in range(maxiter):
            if self.verbose:
                util.dot()
            # FIXME: Would be better to separate regularized from non-regularized part
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
            # Plot matrix
            self.plot(it, show, data, xsol)
        if show:
            plt.show()
        return xsol


def learn(data, **kwargs):
    fgrad = kwargs.get('fgrad', None)
    maxiter = kwargs.get('maxiter', 3000)
    step = kwargs.get('step')
    reg = kwargs.get('reg')
    exact = kwargs.get('exact', False)
    nsamples = kwargs.get('nsamples')
    show = kwargs.get('show', False)
    init_theta = kwargs.get('init_theta', 'diag')
    init_range = kwargs.get('init_range', 0.2)
    verbose = kwargs.get('verbose', True)

    if fgrad is not None:
        opt = AdaOptimizer(fgrad, verbose)
    elif exact:
        opt = AdaOptimizer(lambda t, x: diff.loglik_data_full(t, x)[1], verbose)
    else:
        opt = AdaOptimizer(lambda t, x: diff.loglik_data(t, x, nsamples), verbose)

    if init_theta is None:
        init_theta = np.random.uniform(-init_range, init_range, (data.nitems, data.nitems))
    elif init_theta == 'diag':
        th = np.zeros((data.nitems, data.nitems))
        init_theta = opt.run(data, maxiter=101, step=1, reg=0, xinit=th, only_diag=True, show=False)
        th = np.random.uniform(-init_range, init_range, (data.nitems, data.nitems))
        np.fill_diagonal(th, 0)
        init_theta += th
    theta = opt.run(data, maxiter=maxiter, step=step, reg=reg, xinit=init_theta, show=show)
    return theta


if __name__ == '__main__':
    import datasets
    data = datasets.tcga('gbm', alt=False, mutonly=False)
    data = util.order_data(data, extra=50)
    print(data)
    theta = learn(data, show=True, step=1.0, reg=0.01, nsamples=50)
