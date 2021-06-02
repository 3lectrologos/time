import argparse
import numpy as np
import matplotlib.pylab as plt
import collections
import joblib
import pickle
import learn
import datasets
import sim
import util
from cpp import diff


RES_DIR = 'res_tcga'


def desc(genes):
    return '_'.join(genes)


def run_hazard():
    niter = 10
    #data = datasets.hazard()
    data = get_data(extra=100)
    res = learn.learn(data, show=False, niter=3000, step=1.0, reg=0.01,
                      exact=False, nsamples=50, init_theta='diag', verbose=True)
    #res = joblib.Parallel(n_jobs=niter)(joblib.delayed(learn.learn)(data, show=False, niter=3000, step=0.2, reg=(1.0, 0.1),
    #                                                             exact=False, nsamples=10, init_theta='diag', verbose=True)
    #                                 for i in range(niter))
    #with open('hazard.pcl', 'wb') as fout:
    #    pickle.dump(res, fout)


def plot_hazard():
    with open('hazard.pcl', 'rb') as fin:
        res = pickle.load(fin)
    std = np.std(res, axis=0)
    data = datasets.hazard()
    labels = []
    for label in data.labels:
        if label.endswith('(M)'):
            labels.append(label[:-3])
        else:
            labels.append(label)

    dif = np.amax(res, axis=0) - np.amin(res, axis=0)
    util.plot_matrix(dif,
                     xlabels=labels, ylabels=labels,
                     vmin=0, vmax=5, cmap='Greys', notext=False,
                     axes_fontsize=11, cell_fontsize=9)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig('hazard_dif.pdf', dpi=300)
    plt.gcf().clear()

    for i, r in enumerate(res):
        lik, _ = diff.loglik_data_full(r, data)
        print(f'{i}: {lik}')
        util.plot_matrix(r,
                         xlabels=labels, ylabels=labels,
                         vmin=-4, vmax=4, cmap='PuOr_r', notext=False,
                         axes_fontsize=11, cell_fontsize=9)
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(f'hazard_{i}.pdf', dpi=300)
        plt.gcf().clear()


def plot_big():
    plt.rcParams['axes.linewidth'] = 0.2
    with open(f'{RES_DIR}_010/_150.pcl', 'rb') as fin:
        data, res = pickle.load(fin)

    nmax = 150
    idxs = range(nmax)

    margs = [data.marginals[idx] for idx in idxs]
    comb = zip(idxs, margs)
    comb = sorted(comb, key=lambda x: x[1], reverse=True)
    idxs, _ = zip(*comb)
    labels = [data.labels[x] for x in idxs]

    util.plot_matrix(res[0][np.ix_(idxs, idxs)],
                     xlabels=labels, ylabels=labels,
                     vmin=-4, vmax=4, cmap='PuOr_r', notext=True,
                     axes_fontsize=2)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig('gbm_big.pdf', dpi=300)
    plt.gcf().clear()


def run_exp(genes, size, niter):
    reg = 0.01
    data = get_data(genes, extra=size)
    njobs = min(20, niter)
    res = joblib.Parallel(n_jobs=njobs)(joblib.delayed(learn.learn)(data, show=False, niter=3000, step=1.0, reg=reg,
                                                                    exact=False, nsamples=50, init_theta='diag', verbose=True)
                                     for i in range(niter))
    res = np.array(res)
    with open(f'{RES_DIR}_{int(1000*reg):03d}/{desc(genes)}_{size}.pcl', 'wb') as fout:
        pickle.dump((data, res), fout)


def run_all(genes):
    sizes = [0, 5, 10, 20, 30]
    niter = 20
    for size in sizes:
        run_exp(genes, size, niter)


def get_dir(th, idxs):
    nsamples = 10000
    s, _ = diff.draw(th, nsamples)
    s = sim.marg_seq(s, idxs)
    c = collections.Counter(s)
    i0, i1 = idxs
    c01, c10 = c[(i0, i1)], c[(i1, i0)]
    z = c01 + c10
    return c01/z, c10/z


def plot_real(genes, size):
    with open(f'{RES_DIR}_010/{desc(genes)}_{size}.pcl', 'rb') as fin:
        data, res = pickle.load(fin)
    dif = np.std(res, axis=0)
    idxs = range(min(50, dif.shape[0]))
    dif = dif[np.ix_(idxs, idxs)]
    labels = data.labels
    util.plot_matrix(dif,
                     xlabels=labels, ylabels=labels,
                     vmin=0, vmax=5, cmap='Greys', notext=False,
                     axes_fontsize=8)
    plt.show()


def plot_ratios(genes):
    sizes = []
    mab, mba = [], []
    sab, sba = [], []
    import os
    dirname = f'{RES_DIR}_010'
    allfiles = os.listdir(dirname)
    for fname in allfiles:
        print(fname)
        if fname.startswith(f'{desc(genes)}_'):
            size = int(fname[len(f'{desc(genes)}_'):-4])
            sizes.append(size)
        elif fname.startswith(f'_'):
            size = int(fname[1:-4])
            sizes.append(size)
        else:
            continue
        with open(f'{dirname}/{fname}', 'rb') as fin:
            data, res = pickle.load(fin)
        idxs = data.idx(genes)
        dirs = [get_dir(r, idxs) for r in res]
        ab, ba = zip(*dirs)
        mab.append(np.mean(ab))
        sab.append(2*np.std(ab) / np.sqrt(len(ab)))
        mba.append(np.mean(ba))
        sba.append(2*np.std(ba) / np.sqrt(len(ba)))
    idxs = range(len(sizes))
    comb = sorted(list(zip(idxs, sizes)), key=lambda x: x[1])
    idxs, sizes = zip(*comb)
    idxs = list(idxs)
    mab = np.array(mab)[idxs]
    sab = np.array(sab)[idxs]
    res = np.array([sizes, mab, sab]).T
    np.savetxt(f'/mnt/c/Users/el3ct/Desktop/timepaper/figures/{desc(genes)}.dat', res)
    plt.errorbar(sizes, mab, yerr=sab, fmt='o-', capsize=3, linewidth=2)
    #plt.errorbar(sizes, mba, yerr=sba, fmt='o-', capsize=3, linewidth=2)
    plt.ylim((0, 1))
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["lmodern"],
    })

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--dif', action='store_true')
    args = parser.parse_args()

    #genes = ['EGFR(A)', 'EGFR']
    #genes = ['TP53', 'IDH1']
    #genes = ['MDM2(A)', 'CDK4(A)']
    genes = ['PDGFRA(A)', 'PDGFRA']

    if args.run:
        run_hazard()
        #run_all(genes)
        #run_all([])
    if args.plot:
        plot_ratios(genes)
    elif args.dif:
        #plot_real(genes, size=10)
        plot_big()

    #plot_hazard()
