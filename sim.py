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
    if time:
        return tuple(res), None
    else:
        return tuple(res)


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


if __name__ == '__main__':
    theta = np.array([
        [0, 5],
        [0, -5]
    ])
    ndata = 10000
    data = draw(theta, ndata)
    p0 = len([d for d in data if d == []]) / ndata
    p1 = len([d for d in data if d == [0]]) / ndata
    p2 = len([d for d in data if d == [1]]) / ndata
    p12 = len([d for d in data if d == [0, 1]]) / ndata
    p21 = len([d for d in data if d == [1, 0]]) / ndata

    print(f'p0 = {p0}')
    print(f'p1 = {p1}')
    print(f'p2 = {p2}')
    print(f'p12 = {p12}')
    print(f'p21 = {p21}')
