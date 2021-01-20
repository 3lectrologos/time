import numpy as np
import matplotlib.pylab as plt
import argparse
import simplex_projection
import datasets
from cpp import diff


def plot_mat(theta, ax, labels):
    util.plot_matrix(np.exp(theta), ax, xlabels=labels, ylabels=labels,
                     vmin=0, vmid=1, vmax=20, cmap='PuOr_r')
    plt.pause(0.001)


class Optimizer:
    GAMMA = 0.9
    
    def __init__(self, grad):
        self.grad = grad

    def run(self, data, niter, xinit, step=0.05, reg=50, show=False):
        if show:
            fig, ax = plt.subplots(1, 1, figsize=(11, 11))
            fig.tight_layout()
            plt.gcf().show()
        
        n = xinit.shape[0]
        mom = np.zeros_like(xinit)
        xsol = np.copy(xinit)
        for it in range(niter):
            xsol += self.GAMMA*mom
            g = self.grad(xsol, data)
            #
            lik = loglik_new(base.mat2vec(xsol), None, data)
            print(lik)
            #
            xsol += step*g
            mom = self.GAMMA*mom + step*g
            # L1-projection
            xvec = base.mat2vec(xsol)
            xvec[n+1:] = simplex_projection.euclidean_proj_l1ball(xvec[n+1:], reg)
            xsol = base.vec2mat(xvec, n)
            if show and it % 100 == 0:
                ax.clear()
                plot_mat(xsol, ax, labels=data.labels)
        if show:
            plt.show()
        return xsol


def learn_toy():
    #data = [[]]*1 + [[0]]*2 + [[0, 1]]*4 + [[0, 2]]*3
    #n = 3
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    data = base.read_data()
    labels = ['TP53(M)', 'MDM2(A)', 'MDM4(A)', 'CDKN2A(D)', 'CDK4(A)',
              'NF1(M)', 'IDH1(M)', 'PTEN(M)', 'PTEN(D)', 'EGFR(M)',
              'RB1(D)', 'PDGFRA(A)', 'FAF1(D)', 'SPTA1(M)', 'PIK3CA(M)',
              'OBSCN(M)', 'CNTNAP2(M)', 'PAOX(M)', 'TP53(D)', 'LRP2(M)']
     
    data = data.subset(data.idx(labels))

    #data = datasets.comet('gbm')
    #cutoff = 0.04
    #keep = []
    #for idx in range(data.nitems):
    #    if data.marginals[idx] > cutoff:
    #        keep.append(idx)
    #data = data.subset(keep)
    #print(data)

    opt = Optimizer(lambda t, x: diff.loglik_data(t, x, 200))
    theta = np.random.uniform(0, 0, (data.nitems, data.nitems))
    theta = opt.run(data, niter=500, xinit=theta, show=args.show)
    print('theta =\n', theta)
    #base.sample_stat(theta, 10000, seq=False)


if __name__ == '__main__':
    learn_toy()
