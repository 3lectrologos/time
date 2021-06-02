import argparse
import collections
import joblib
import pickle
import os
import multiprocessing

import numpy as np
import matplotlib.pylab as plt

import learn
import datasets
import util
from cpp import diff


DIR_NAME = 'result_gbm'


def desc(genes):
    return '_'.join(genes)


def run_hazard():
    #nreps = 10
    data = datasets.hazard()
    data = util.order_data(data, extra=100)
    res = learn.learn(data, reg=0.01, nsamples=50)
    #res = joblib.Parallel(n_jobs=nreps)(joblib.delayed(learn.learn)(data, show=False, niter=3000, step=0.2, reg=(1.0, 0.1),
    #                                                                exact=False, nsamples=10, init_theta='diag', verbose=True)
    #                                 for i in range(nreps))
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
    with open(f'{DIR_NAME}/_150.pcl', 'rb') as fin:
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


def plot_real(genes, size):
    with open(f'{DIR_NAME}/{desc(genes)}_{size}.pcl', 'rb') as fin:
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


def run_one(genes, size, nreps):
    data = datasets.tcga('gbm')
    data = util.order_data(data, fixed=genes, extra=size)
    njobs = multiprocessing.cpu_count()
    res = joblib.Parallel(n_jobs=njobs)(
        joblib.delayed(learn.learn)(data, reg=0.01, nsamples=50)
        for i in range(nreps))
    res = np.array(res)
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)
    with open(f'{DIR_NAME}/{desc(genes)}_{size}.pcl', 'wb') as fout:
        pickle.dump((data, res), fout)


def run(genes, sizes, nreps):
    omp_threads = os.environ['OMP_NUM_THREADS']
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        for size in sizes:
            run_one(genes, size, nreps)
    finally:
        os.environ['OMP_NUM_THREADS'] = omp_threads


def run_all(nreps):
    gene_pairs = (
        ['EGFR(A)', 'EGFR'],
        ['PDGFRA(A)', 'PDGFRA'],
        ['TP53', 'IDH1'],
        ['MDM2(A)', 'CDK4(A)']
    )
    pair_sizes = [0, 5, 10, 20, 30]
    for pair in gene_pairs:
        run(pair, pair_sizes, nreps)
    other_sizes = [50, 70, 100, 150, 200]
    run([], other_sizes, nreps)


def get_direction(th, idxs):
    nsamples = 100000
    s, _ = diff.draw(th, nsamples)
    s = util.marg_seq(s, idxs)
    c = collections.Counter(s)
    i0, i1 = idxs
    c01, c10 = c[(i0, i1)], c[(i1, i0)]
    z = c01 + c10
    return c01/z, c10/z


def plot_direction(genes):
    sizes = []
    mab, mba = [], []
    sab, sba = [], []
    import os
    allfiles = os.listdir(DIR_NAME)
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
        with open(f'{DIR_NAME}/{fname}', 'rb') as fin:
            data, res = pickle.load(fin)
        idxs = data.idx(genes)
        dirs = [get_direction(r, idxs) for r in res]
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
    np.savetxt(f'{DIR_NAME}/{desc(genes)}.dat', res)
    plt.errorbar(sizes, mab, yerr=sab, fmt='o-', capsize=3, linewidth=2)
    plt.ylim((0, 1))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', nargs=1)
    parser.add_argument('--plot', nargs=2)
    args = parser.parse_args()

    if args.run:
        run_all(int(args.run[0]))
    if args.plot:
        g1, g2 = args.plot[0], args.plot[1]
        plot_direction([g1, g2])
