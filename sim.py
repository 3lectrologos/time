import numpy as np
import scipy.special
import collections


def mat(qs):
    q1, q2, q12, q21 = qs
    Q = np.array([
        [-q1-q2, q1,      q2,      0],
        [0,      -q2*q12, 0,       q2*q12],
        [0,      0,       -q1*q21, q1*q21],
        [0,      0,       0,       0]
    ])
    return Q


def probs(qs):
    Q = mat(qs)
    R = np.linalg.inv(np.eye(4) - Q)
    p0 = np.array([1, 0, 0, 0])
    return p0.dot(R)


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
    return tuple(res)


def draw_one(th, time=False):
    ground = set(range(th.shape[0]))
    tstop = np.random.exponential(1.0)
    res = []
    tcur = 0
    while True:
        sumth = []
        rest = list(ground - set(res))
        if rest == []:
            break
        for j in rest:
            sumth.append(th[j, j] + sum(th[i, j] for i in res))
        sumth = np.asarray(sumth + [0])
        lse = scipy.special.logsumexp(sumth)
        dt = np.random.exponential(1.0/np.exp(lse))
        tcur += dt
        if tcur > tstop:
            break
        ps = np.exp(sumth - lse)
        next = np.random.choice(rest + [-1], 1, p=ps)[0]
        res.append(next)
    if time:
        return tuple(res), tstop
    else:
        return tuple(res)


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


if __name__ == '__main__':
    q1 = 0.5
    q2 = 0.01
    q12 = 50
    q21 = 1
    qs = (q1, q2, q12, q21)
    p = probs(qs)
    print(f'p = {p}')

    nsamples = 10000
    samples = [draw(mat(qs)) for i in range(nsamples)]
    #samples = [s[-1] for s in samples]
    counter = collections.Counter(samples)
    for s in counter:
        counter[s] /= nsamples
    print(counter)


    q1 = 0.38
    q2 = 0.13
    q12 = 1
    q21 = 50
    qs = (q1, q2, q12, q21)
    p = probs((q1, q2, q12, q21))
    print(f'p = {p}')

    nsamples = 10000
    samples = [draw(mat(qs)) for i in range(nsamples)]
    #samples = [s[-1] for s in samples]
    counter = collections.Counter(samples)
    for s in counter:
        counter[s] /= nsamples
    print(counter)
