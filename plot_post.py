import numpy as np
import matplotlib.pylab as plt
import random
import scipy.stats
import scipy.linalg
import poibin
import networkx as nx
import joblib
import learn
from cpp import diff, tdiff


def fprob(t, k, th):
    pb = poibin.PoiBin(1-np.exp(-th*t))
    return pb.pmf(k) * np.exp(-t)


def var(k, n, logth):
    th = np.exp(logth)
    f = lambda t: scipy.stats.binom.pmf(k, n, 1-np.exp(-th*t)) * np.exp(-t)
    norm = scipy.integrate.quad(f, 0, 50)[0]
    ft = lambda t: t*f(t)
    ft2 = lambda t: t*t*f(t)
    m = scipy.integrate.quad(ft, 0, 50)[0] / norm
    m2 = scipy.integrate.quad(ft2, 0, 50)[0] / norm
    return m2 - m*m


def plot_var(logth):
    ks = 9*np.array(range(1, 40, 2))
    ns = 10*np.array(range(1, 40, 2))
    vars = [var(k, n, logth) for k, n in zip(ks, ns)]
    plt.loglog(ns, vars, '-o')


def plot_dot(r, logth):
    ns = 10.0*np.array(range(1, 40, 2))
    f = r*(1/np.exp(logth)**2)*(1/ns)
    plt.loglog(ns, f, 'r--')


def plot_vars():
    plot_var(0)
    plot_var(-1)
    plot_var(-2)
    #ns = 10.0*np.array(range(35))
    #f = 0.25*ns**(-1)
    #plt.plot(ns, f, 'k--')
    #f = 7*ns**(-1)
    #plt.plot(ns, f, 'k--')
    r = 9
    plot_dot(r, 0)
    plot_dot(r, -1)
    plot_dot(r, -2)
    plt.show()

    
def getp(ts, k, logth):
    th = np.exp(logth)
    #p = np.exp(-np.square(s-q) / (2*q*(1-q)/n)) * np.exp(-t)
    #f = lambda t: scipy.stats.binom.pmf(k, n, 1-np.exp(-th*t)) * np.exp(-t)
    f = lambda t: fprob(t, k, th)
    norm = scipy.integrate.quad(f, 0, 20)[0]
    p = np.array([f(t) for t in ts])
    p /= norm
    return p


def fprob_general(theta, tind, sample, k, t):
    pb = poibin.PoiBin(1-np.exp(-np.exp(tind)*t))
    return pb.pmf(k) * np.exp(tdiff.loglik_set(theta, sample, t)[0] - t)


def getp_general(theta, tind, sample, k, ts):
    tmax = 10
    f = lambda t: fprob_general(theta, tind, sample, k, t)
    norm = scipy.integrate.quad(f, 0, tmax)[0]
    p = np.array([f(t) for t in ts])
    p /= norm
    return p


def test_getp_general():
    theta = np.array([[0, 3],
                      [0, -3]])
    ndep = 2
    nrest = 100
    tind = np.random.uniform(-1, -1, nrest)
    sample = [0, 1]

    ts = np.linspace(0, 4, 1000)
    ks = [10, 30, 60]
    pmax = 0
    plt.figure(figsize=(3, 2), dpi=300)
    for k in ks:
        ps = getp_general(theta, tind, sample, k, ts)
        pmax = max(np.max(ps), pmax)
        plt.plot(ts, ps, '-', linewidth=1.5)
    plt.ylim((0, 1.05*pmax))
    plt.gca().get_yaxis().set_ticks([])
    plt.show()


def plot_example():
    lth = -2
    
    t = np.linspace(0, 3, 10000)
    p0 = np.exp(-t)
    p1 = getp(t, 1, [lth]*10)
    p2 = getp(t, 10, [lth]*100)
    #p3 = getp(t, 30, [lth]*1000)
    #t4, p4 = getp(s, 10000, lth)
    plt.plot(t, p0, t, p1, t, p2, linewidth=2)
    plt.show()


def plot_data(keep=0, seed=0):
    alpha = 0.02
    xmax = 5
    dataidx = 0
    
    plt.gca().clear()
    data, times, theta, ndep = learn.get_data(dataidx)
    choices = list(range(ndep, data.nitems))
    random.seed(seed)
    random.shuffle(choices)
    ind = choices[:keep]
    #ps = np.diag(theta)[ndep:ndep+keep]
    #data = data.subset(list(range(ndep+keep)))
    ps = np.diag(theta)[ind]
    data = data.subset(list(range(ndep)) + ind)
    n = data.nitems
    data = [data[i] for i in range(1000)]
    x = np.linspace(0, xmax, 300)
    reds = [x]
    blues = [x]
    for idata, d in enumerate(data):
        print(idata)
        rest = len([i for i in d if i not in list(range(ndep))])
        y = getp(x, rest, ps)
        #y = np.exp(-x)
        #if 0 not in d and 1 not in d:
        #    plt.plot(x, y, 'g-', alpha=0.5)
        if (0 in d) and (1 not in d):
            if times[idata] > 1.1 and times[idata] < 1.2:
                print('====>', idata)
            plt.fill_between(x, 0, y, color='#ff360e', alpha=alpha)
            reds.append(y)
        elif (0 in d) and (1 in d):
            plt.fill_between(x, 0, y, color='#1f77b4', alpha=alpha)
            blues.append(y)
    #
    idx = 382
    d = data[idx]
    print('TIME:', times[idx])
    rest = len([i for i in d if i not in list(range(ndep))])
    x = np.linspace(0, xmax, 500)
    y = getp(x, rest, ps)
    plt.plot(x, y, 'k--', linewidth=2.5)
    plt.plot(times[idx], 0, 'ks', markersize=8, clip_on=False)
    #
    np.savetxt(f'/mnt/c/Users/el3ct/Desktop/timepaper/figures/reds_{keep}.dat', np.array(reds).T)
    np.savetxt(f'/mnt/c/Users/el3ct/Desktop/timepaper/figures/blues_{keep}.dat', np.array(blues).T)
    plt.ylim((0, 2.5))
    plt.xlim((0, xmax))
    plt.xlabel('$t$', fontsize=18, labelpad=-5)
    plt.xticks(fontsize=16)
    plt.gca().get_yaxis().set_ticks([])
    plt.gcf().set_size_inches(5, 3.5)
    plt.tight_layout()
    plt.savefig(f'post_{keep}', dpi=300, interpolation='antialiased')


def comp_prob(theta, comps, rest, sample, t):
    # Prior
    prob = -t
    # Components
    for comp in comps:
        thcomp = theta[np.ix_(comp, comp)]
        scomp = []
        k = 0
        for c in comp:
            if c in sample:
                scomp.append(k)
            k += 1
        prob += tdiff.loglik_set(thcomp, scomp, t)[0]
    # Rest
    tind = np.diag(theta)[rest]
    k = len(set(rest) & set(sample))
    pb = poibin.PoiBin(1-np.exp(-np.exp(tind)*t))
    prest = pb.pmf(k)
    # Result
    return prest * np.exp(prob)
    

def prob_by_components(theta, sample, ts):
    n = theta.shape[0]
    g = nx.DiGraph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(n):
            if theta[i, j] != 0:
                g.add_edge(i, j)
    cc = nx.weakly_connected_components(g)
    comps = []
    rest = []
    for c in cc:
        if len(c) > 1:
            comps.append([i for i in c])
        else:
            rest += list(c)

    f = lambda t: comp_prob(theta, comps, rest, sample, t)
    norm = scipy.integrate.quad(f, 0, 50)[0]
    #p = np.array([f(t) for t in ts])
    #p /= norm
    ft = lambda t: t*f(t)
    ft2 = lambda t: t*t*f(t)
    m = scipy.integrate.quad(ft, 0, 50)[0] / norm
    m2 = scipy.integrate.quad(ft2, 0, 50)[0] / norm
    var = m2 - m*m
    return var


def get_freqs(th):
    freqs = []
    data, _ = diff.draw(th, 100000)
    for i in range(th.shape[0]):
        fi = len([d for d in data if i in d]) / len(data)
        freqs.append(fi)
    return freqs


def match_freqs(th, freqs):
    newth = th.copy()
    n = th.shape[0]
    niter = 50
    step = 1.0
    for it in range(niter):
        data, _ = diff.draw(newth, 10000)
        ds = []
        for i in range(n):
            di = len([d for d in data if i in d]) / len(data)
            newth[i, i] += (freqs[i] - di)*step
    return newth


def get_one_m(m, data, blocks, ts):
    print(f'Running {m}')
    mvar = []
    for i, dat in enumerate(data):
        print(i)
        #keep = np.random.choice(list(range(nrest)), size=m, replace=False)
        keep = list(range(m))
        random.shuffle(blocks)
        ttmp = scipy.linalg.block_diag(*blocks)
        tkeep = ttmp[np.ix_(keep, keep)]
        var = prob_by_components(tkeep, dat, ts)
        mvar.append(var)
    return mvar


def plot_one(tind, ms, strength, ndata):
    blocksize = 2
    nblocks = 50
    blocks = []
    for b in range(nblocks):
        print('block', b)
        #tblock = np.zeros((blocksize, blocksize))
        idxs = list(range(b*blocksize, (b+1)*blocksize))
        tblock = np.diag(tind[idxs])
        for i in range(blocksize):
            for j in range(blocksize):
                if i != j and np.random.uniform() < 1.0:
                    val = np.random.uniform(strength, strength)
                    tblock[i, j] = val
        #print(tblock)
        idxs = list(range(b*blocksize, (b+1)*blocksize))
        #print(trest[np.ix_(idxs, idxs)])
        freqs = get_freqs(np.diag(tind[idxs]))
        print('desired:', freqs)
        tnew = match_freqs(tblock, freqs)
        freqs = get_freqs(tnew)
        print(tnew)
        print('obtained:', freqs)
        print('------------')
        blocks.append(tnew.copy())

    tfinal = scipy.linalg.block_diag(*blocks)
    tdep = np.array([[0, 4], [0, -4]])
    tfull = scipy.linalg.block_diag(tdep, tfinal)
    data, _ = diff.draw(tfull, ndata)
    data = [d for d in data if (0 in d) or (1 in d)]

    ts = np.linspace(0, 5, 100)
    vars = joblib.Parallel(n_jobs=20)(joblib.delayed(get_one_m)(m, data, blocks, ts)
                                      for m in ms)
    vars = np.array(vars)
    meanvars = np.mean(vars, axis=1)
    stdvars = 2*np.std(vars, axis=1) / np.sqrt(len(data))
    plt.errorbar(ms, meanvars, yerr=stdvars, fmt='o-', capsize=3, linewidth=2)
    return meanvars, stdvars


def plot_ind():
    ms = [0, 5, 10, 20, 30, 40, 50, 60, 80, 100]
    nrest = 100
    ndata = 1000

    tind = np.random.uniform(-3, -3, nrest)
    m1, s1 = plot_one(tind, ms, strength=0, ndata=ndata)
    tind = np.random.uniform(-2, -2, nrest)
    m2, s2 = plot_one(tind, ms, strength=0, ndata=ndata)
    tind = np.random.uniform(-1, -1, nrest)
    m3, s3 = plot_one(tind, ms, strength=0, ndata=ndata)
    tind = np.random.uniform(-3, -1, nrest)
    m4, s4 = plot_one(tind, ms, strength=0, ndata=ndata)

    means = np.array([m1, m2, m3, m4])
    stds = np.array([s1, s2, s3, s4])
    res = np.hstack((np.atleast_2d(ms).T, means.T, stds.T))
    np.savetxt(f'/mnt/c/Users/el3ct/Desktop/timepaper/figures/two_ind.dat', res)
    
    #plt.ylim((0, 1.5))
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.show()


def plot_dep():
    ms = [0, 5, 10, 20, 30, 40, 50, 60, 80, 100]
    nrest = 100
    ndata = 1000

    means = []
    stds = []
    tind = np.random.uniform(-2, -2, nrest)
    for st in [0, 4, 8, 12]:
        m, s = plot_one(tind, ms, strength=-st, ndata=ndata)
        means.append(m)
        stds.append(s)
    means = np.array(means)
    stds = np.array(stds)
    res = np.hstack((np.atleast_2d(ms).T, means.T, stds.T))
    np.savetxt(f'/mnt/c/Users/el3ct/Desktop/timepaper/figures/two_rep.dat', res)
    
    #plt.ylim((0, 1.5))
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.show()


if __name__ == '__main__':
    #plot_example()
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["lmodern"],
    })

    #ms = [0, 5, 25, 100]
    #for m in ms:
    #    print(f'Plotting {m}')
    #    plot_data(m)

    #plot_vars()
    plot_dep()
