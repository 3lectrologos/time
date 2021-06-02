import numpy as np
import joblib
import pickle
import matplotlib.pylab as plt
import datasets
from cpp import diff
import learn


def get_data():
    data = datasets.tcga('gbm', alt=False, mutonly=False)
    cutoff = 0.035
    keep = []
    for idx in range(data.nitems):
        if data.marginals[idx] > cutoff:
            keep.append(idx)
    data = data.subset(keep)
    return data


def test(theta, data, fgrad, truegrad):
    truenorm = np.linalg.norm(truegrad)
    print('NORM = ', truenorm)
    dif = np.abs(grad-truegrad)
    reldif = np.abs(grad-truegrad) / np.abs(truegrad)
    print('MAX DIF =', np.max(dif))
    print('MEAN DIF =', np.mean(dif))
    print('MAX REL DIF =', np.max(reldif))
    print('MEAN REL DIF =', np.mean(reldif))
    return np.max(dif)
    #return np.linalg.norm(dif)


def check_one(i, data, ftheta, nsamples):
    #print('====', i)
    np.random.seed(i)
    theta = ftheta(i, data.nitems)
    
    f1 = lambda t, d: diff.loglik_data(t, d, nsamples)
    f2 = lambda t, d: diff.loglik_data_uniform(t, d, nsamples)
    #ferror = lambda g, t: np.max(np.abs(g-t))
    #keep = list(range(2))
    #ferror = lambda g, t: np.linalg.norm((g-t)[np.ix_(keep, keep)])
    ferror = lambda g, t: np.linalg.norm(g-t)
    
    _, truegrad = diff.loglik_data_full(theta, data)
    print(truegrad.shape)
    grad = f1(theta, data)
    eprop = ferror(grad, truegrad)
    grad = f2(theta, data)
    eunif = ferror(grad, truegrad)
    return eprop, eunif


if __name__ == '__main__':
    niter = 100
    size = 20
    nsamples_list = [5, 10, 20, 30, 40, 50]
    #fulldata = learn.get_real()
    #ftheta = lambda i, n: np.random.uniform(-1, 1, (n, n))
    with open('tmptheta.pcl', 'rb') as fin:
        data, theta = pickle.load(fin)
    keep = list(range(size))
    ftheta = lambda i, n: theta[np.ix_(keep, keep)]
    data = data.subset(keep)

    umeans = []
    ustds = []
    pmeans = []
    pstds = []
    for nsamples in nsamples_list:
        print('nsamples =', nsamples)
        #data = fulldata.subset(list(range(size)))
        res = joblib.Parallel(n_jobs=20)(joblib.delayed(check_one)(i, data, ftheta, nsamples)
                                         for i in range(niter))
        rprop, runif = zip(*res)
        umean = np.mean(runif)
        ustd = 2*np.std(runif) / np.sqrt(niter)
        pmean = np.mean(rprop)
        pstd = 2*np.std(rprop) / np.sqrt(niter)
        print(f'Unif: {umean} +/- {ustd}')
        print(f'Prop: {pmean} +/- {pstd}')
        umeans.append(umean)
        ustds.append(ustd)
        pmeans.append(pmean)
        pstds.append(pstd)
    res = np.array([nsamples_list, umeans, ustds, pmeans, pstds]).T
    np.savetxt(f'/mnt/c/Users/el3ct/Desktop/timepaper/figures/proposal_{size}.dat', res)
    plt.errorbar(nsamples_list, umeans, yerr=ustds, fmt='o-', capsize=3, linewidth=2)
    plt.errorbar(nsamples_list, pmeans, yerr=pstds, fmt='o-', capsize=3, linewidth=2)
    plt.show()
