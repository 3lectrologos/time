import numpy as np
import scipy.special
import collections


def draw_old(Q):
    n = Q.shape[0]
    P = np.eye(n) - np.diag(1.0/np.diag(Q)) @ Q
    tstop = np.random.exponential(1.0)
    tcur = 0
    scur = 0
    ss = []
    while tcur < tstop:
        ss.append(scur)
        lam = -Q[scur, scur]
        if lam == 0:
            break
        dt = np.random.exponential(1.0/lam)
        tcur += dt
        scur = np.random.choice(list(range(n)), p=P[scur, :])
    return tuple(ss)


def draw_one_old(th, time=False):
    ground = set(range(th.shape[0]))
    res = []
    while True:
        sumth = []
        rest = list(ground - set(res))
        for j in rest:
            sumth.append(th[j, j] + sum(th[i, j] for i in res))
        sumth = np.asarray(sumth + [0])
        lse = scipy.special.logsumexp(sumth)
        ps = np.exp(sumth - lse)
        next = np.random.choice(rest + [-1], 1, p=ps)[0]
        if next == -1:
            break
        else:
            res.append(next)
    if time:
        return res, None
    else:
        return res


def draw_one(th, time=False):
    ground = set(range(th.shape[0]))
    # XXX
    tstop = np.random.exponential(1.0)
    #tstop = 0.1
    #coin = np.random.random()
    #if coin < 0.5:
    #    tstop = 1
    #else:
    #    tstop = 2
    res = []
    tcur = 0
    while True:
        rest = list(ground - set(res))
        if rest == []:
            break
        sumth = np.array([th[j, j] + sum(th[i, j] for i in res)
                          for j in rest])
        lse = scipy.special.logsumexp(sumth)
        dt = np.random.exponential(1.0/np.exp(lse))
        tcur += dt
        if tcur > tstop:
            break
        ps = np.exp(sumth - lse)
        #print(f'res = {res}, rest = {rest}, ps = {ps}')
        next = np.random.choice(rest, 1, p=ps)[0]
        res.append(next)
    if time:
        return res, tstop
    else:
        return res


def draw(th, nsamples=None, time=False):
    if nsamples is None:
        return draw_one(th, time)
    else:
        res = [draw_one(th, time) for i in range(nsamples)]
        if time:
            samples, times = zip(*res)
            return list(samples), list(times)
        else:
            return res


def marg_seq(ss, idxs):
    ssnew = []
    for s in ss:
        snew = [x for x in s if x in idxs]
        ssnew.append(snew)
    return ssnew


def tv_seq(th1, th2, ndep, nsamples):
    s1 = draw(th1, nsamples)
    s2 = draw(th2, nsamples)
    s1 = marg_seq(s1, list(range(ndep)))
    s2 = marg_seq(s2, list(range(ndep)))
    s1 = [tuple(s) for s in s1]
    s2 = [tuple(s) for s in s2]

    c1 = collections.Counter(s1)
    c2 = collections.Counter(s2)
    tv = 0
    ks, p1, p2 = [], [], []
    for k in set(c1.keys()) | set(c2.keys()):
        ks.append(k)
        p1.append(c1[k]/nsamples)
        p2.append(c2[k]/nsamples)
    print(ks)
    print(p1)
    print(p2)
    import scipy.stats
    #return 0.5*np.sum(np.abs(np.array(p1)-np.array(p2)))
    return scipy.stats.entropy(p2, p1)


if __name__ == '__main__':
    th1 = np.array([
        [0, 3, 0],
        [0, -3, 0],
        [0, 0, -2]
    ])

    th2 = np.array([
        [0, 2, 0],
        [1, -1, 0],
        [0, 0, -2]
    ])

    kl = tv_seq(th1, th1, ndep=2, nsamples=10000)
    print(kl)
