import os
import shutil
import argparse
import pickle
import collections
import random
import multiprocessing

import joblib
import numpy as np
import matplotlib.pylab as plt

import datasets
from cpp import diff, tdiff
import learn
import util


DATA_DIR = 'datasets/synth'
OUT_DIR = 'result_synth'


def synth2():
    theta = np.array([
        [0, 4],
        [0, -4]
    ])
    return theta


def synth5():
    theta = np.array([
        [-1, 4, 0, 0, 0],
        [0, -1, -2, -2, 0],
        [0, -2, -1, -2, 0],
        [0, -2, -2, -0.5, 4],
        [0, 0, 0, 0, -4]
    ])
    return theta


def get_theta(ftheta, nrest, uni_lo, uni_hi):
    theta = ftheta()
    ndep = theta.shape[0]
    tind = np.random.uniform(uni_lo, uni_hi, nrest)
    trest = np.diag(tind)
    theta = np.block([[theta, np.zeros((ndep, nrest))],
                      [np.zeros((nrest, ndep)), trest]])
    return theta, ndep


def draw_data(ftheta, ndata=500, nrest=100, nreps=1):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    theta, ndep = get_theta(nrest)
    for i in range(nreps):
        lst, times = diff.draw(theta, ndata)
        for s in lst:
            random.shuffle(s)
        with open(f'{DATA_DIR}/{ftheta.__name__}_{i}.pcl', 'wb') as fout:
            pickle.dump((lst, times, ndep, nrest, theta), fout)


def get_data(ftheta, i=0):
    with open(f'{DATA_DIR}/{ftheta.__name__}_{i}.pcl', 'rb') as fin:
        lst, times, ndep, nrest, true = pickle.load(fin)
    data = datasets.Data.from_list([lst], nitems=ndep+nrest)
    return data, times, true, ndep
    

def learn_time(ftheta):
    data, times, truetheta, ndep = get_data(ftheta)
    data = data.subset(list(range(ndep)))
    fgrad = lambda t, x: tdiff.loglik_data(t, x, times)[1]
    theta = learn.learn(data, fgrad=fgrad, show=args.show, maxiter=3000, step=1.0, reg=0.01,
                        exact=True, nsamples=50, verbose=True)
    print('\nLearned theta =')
    print(theta)
    truetheta = truetheta[np.ix_(list(range(ndep)), list(range(ndep)))]
    dif = util.KLdist(truetheta, theta, ndep, nsamples=100000)
    print('Dif =', dif)
    return dif


def recover_one(ftheta, size, rep, feval=None):
    print(f'Running {ftheta.__name__} | size: {size} -- rep: {rep}')
    data, _, truetheta, ndep = get_data(ftheta)
    choices = list(range(ndep, data.nitems))
    random.seed(rep)
    random.shuffle(choices)
    ind = choices[:size]
    data = data.subset(list(range(ndep)) + list(ind))
    theta = learn.learn(data, show=args.show, maxiter=3000, step=1.0, reg=0.01,
                        exact=False, nsamples=50, verbose=True)
    tt = truetheta[np.ix_(list(range(ndep)), list(range(ndep)))]
    dif = feval(tt, theta, ndep)
    print(f'Done {ftheta.__name__} | size: {size} -- rep: {rep}')
    omp_threads = os.environ['OMP_NUM_THREADS']
    return data, ndep, truetheta, theta, dif


Result = collections.namedtuple('Result', ['data', 'ndep', 'truetheta', 'theta', 'dif'])


def recover(ftheta, nreps):
    try:
        omp_threads = os.environ['OMP_NUM_THREADS']
    except KeyError:
        omp_threads = None
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        feval = lambda t1, t2, ndep: util.KLdist(t1, t2, ndep, nsamples=1000000)
        njobs = multiprocessing.cpu_count()
        res = joblib.Parallel(n_jobs=njobs)(joblib.delayed(recover_one)(ftheta, size, rep, feval)
                                            for size in sizes
                                            for rep in range(nreps))
        res = [Result(*r) for r in res]
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
        with open(f'{OUT_DIR}/result_{ftheta.__name__}.pcl', 'wb') as fout:
            pickle.dump((sizes, nreps, res), fout)
    finally:
        if omp_threads is not None:
            os.environ['OMP_NUM_THREADS'] = omp_threads


def plot(ftheta):
    timedif = learn_time(ftheta)
    with open(f'{OUT_DIR}/result_{ftheta.__name__}.pcl', 'rb') as fin:
        sizes, nreps, res = pickle.load(fin)
    difs = [r.dif for r in res]
    difs = np.array(difs).reshape(len(sizes), nreps)
    means = np.mean(difs, axis=1)
    stds = 2*np.std(difs, axis=1) / np.sqrt(difs.shape[1])
    out = np.array([sizes, means, stds]).T
    np.savetxt(f'{ftheta.__name__}.dat', out)
    plt.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=3, linewidth=2)
    plt.plot([0, max(sizes)], [timedif, timedif], 'k--')
    plt.xlim((-0.5, sizes[-1]+0.5))
    plt.ylim((0, 0.5))
    plt.xlabel(r'm', fontsize=18)
    plt.ylabel(r'KL div.', fontsize=18)
    plt.gca().tick_params(labelsize=14)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--run', nargs=1)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('name')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    ftheta = locals()[args.name]
    if args.draw:
        draw_data(ftheta)
    elif args.time:
        learn_time(ftheta)
    elif args.run:
        recover(ftheta, int(args.run[0]))
    elif args.plot:
        plot(ftheta)
