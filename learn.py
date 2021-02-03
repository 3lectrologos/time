import numpy as np
import matplotlib.pylab as plt
import argparse
import simplex_projection
import datasets
import pydiff
import util
from cpp import diff


def plot_mat(theta, ax, labels):
    thresh = 0.1
    n = theta.shape[0]
    mat = np.abs(theta)
    np.fill_diagonal(mat, 0)
    idxs = [i for i in range(n)
            if (np.max(mat[i, :]) > thresh or np.max(mat[:, i]) > thresh)]
    if idxs == []:
        plt.pause(0.001)
        return
    idxlab = list(zip(idxs, [labels[i] for i in idxs]))
    idxlab = sorted(idxlab, key=lambda x: x[1])
    idxs, labs = zip(*idxlab)
    print([labels[x] for x in idxs])
    util.plot_matrix(np.exp(theta), ax,
                     xlabels=labels, ylabels=labels, permx=list(idxs), permy=list(idxs),
                     vmin=0, vmid=1, vmax=5, cmap='PuOr_r', notext=False)
    plt.pause(0.001)


class Optimizer:
    GAMMA = 0.9
    
    def __init__(self, grad):
        self.grad = grad

    def run(self, data, niter, xinit, reg, step=0.2, show=False):
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
            xsol += step*g
            mom = self.GAMMA*mom + step*g
            # L1-projection
            # TODO: Is this slow?
            xvec = util.mat2vec(xsol)
            xvec[n:] = simplex_projection.euclidean_proj_l1ball(xvec[n:], reg)
            xsol = util.vec2mat(xvec, n)
            if it % 50 == 49: util.dot10()
            if show and it % 50 == 0:
                ax.clear()
                plot_mat(xsol, ax, labels=data.labels)
        #
        lik = diff.loglik_data_full(xsol, data)
        print('log-lik =', lik[0])
        #
        if show:
            plt.show()
        return xsol


def learn():
    #data = [[]]*1 + [[0]]*2 + [[0, 1]]*4 + [[0, 2]]*3
    #n = 3
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    #data = datasets.hazard()
    #labels = ['TP53(M)', 'MDM2(A)', 'MDM4(A)', 'CDKN2A(D)', 'CDK4(A)',
    #          'NF1(M)', 'IDH1(M)', 'PTEN(M)', 'PTEN(D)', 'EGFR(M)',
    #          'RB1(D)', 'PDGFRA(A)', 'FAF1(D)', 'SPTA1(M)', 'PIK3CA(M)',
    #          'OBSCN(M)', 'CNTNAP2(M)', 'PAOX(M)', 'TP53(D)', 'LRP2(M)']
    #data = data.subset(data.idx(labels))

    data = datasets.linear_2(nsamples=10000)
    #print([data[i] for i in range(50)])
    #data = datasets.ind_2(nsamples=300)

    if False:
        import tcga
        data = tcga.data('coadreadnew', alt=False, mutonly=True)
        if False:
            cutoff = 0.0
            keep = []
            for idx in range(data.nitems):
                if data.marginals[idx] > cutoff:
                    keep.append(idx)
            print([data.labels[x] for x in keep])
        else:
            labels = ['KRAS', 'APC']
            keep = data.idx(labels)
        data = data.subset(keep)
        print(data)

    opt = Optimizer(lambda t, x: diff.loglik_data_full(t, x)[1])
    theta = np.random.uniform(0, 0, (data.nitems, data.nitems))
    theta = opt.run(data, niter=1000, reg=1, xinit=theta, show=args.show)
    #print('theta =\n', theta)
    import base
    base.sample_stat(theta, 10000, seq=True)


if __name__ == '__main__':
    learn()
