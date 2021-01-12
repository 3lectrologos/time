import numpy as np
import itertools
import scipy.special
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax
import lbfgs
import base
from datasets import Data


def q(th, s, a):
    res = th[a, a]
    for i in s:
        res += th[i, a]
    return res


def loglik_seq(th, x):
    n = th.shape[0]
    res = 0
    i = 0
    for i in range(len(x)):
        num = q(th, x[:i], x[i])
        rest = [q(th, x[:i], j)
                for j in set(range(n)) - set(x[:i+1])]
        res += (num - jsp.logsumexp(jnp.array([0, num] + rest)))
    # Final factor
    rest = [q(th, x[:i+1], j)
            for j in set(range(n)) - set(x[:i+1])]
    res -= jsp.logsumexp(jnp.array([0] + rest))
    return res


def loglik_set(th, x):
    lls = [loglik_seq(th, xperm) for xperm in itertools.permutations(x)]
    return jsp.logsumexp(jnp.array(lls))


def loglik(thvec, gout, data, gfun):
    n = int(np.sqrt(thvec.shape[0]))
    th = jnp.array(base.vec2mat(thvec, n))
    val = 0
    grd = np.zeros((n, n))
    for d in data:
        v, g = gfun(th, d)
        val += v
        grd += g
    grd = base.mat2vec(grd)
    if gout is not None:
        for i in range(len(gout)):
            gout[i] = -grd[i]/len(data)
    return -val/len(data)


def learn():
    gfun = jax.value_and_grad(loglik_set, argnums=0)
    n = 3
    data = [[]]*1 + [[0]]*2 + [[0, 1]]*4 + [[0, 2]]*3
    
    fopt = lambda x, g: loglik(x, g, data, gfun)
    x0 = np.random.uniform(-5, 5, (n, n))
    sol = lbfgs.fmin_lbfgs(fopt, x0=base.mat2vec(x0), epsilon=1e-8)
    #orthantwise_c = 0.01
    #sol = lbfgs.fmin_lbfgs(fopt, x0=base.mat2vec(x0), epsilon=1e-8,
    #                       orthantwise_c=orthantwise_c,
    #                       orthantwise_start=n,
    #                       line_search='wolfe')

    lik = -loglik(sol, None, data, gfun)
    print('loglik =', lik)
    sol = base.vec2mat(sol, n)
    print(sol)
    #print('exp(sol) =')
    #print(np.exp(sol))
    base.sample_stat(sol, 10000, seq=False)


def test():
    n = 4
    data = [[0], [2, 3, 1], [3, 0]]
    gfun = jax.value_and_grad(loglik_set, argnums=0)
    theta = np.random.uniform(-5, 5, (n, n))

    g = np.ones(n*n)
    val = loglik(base.mat2vec(theta), g, data, gfun)
    print('val =', val)
    print('grd =')
    print(base.vec2mat(g, n))

    data = Data.from_list([data], nitems=n)
    pdata = base.get_pdata(data)
    g = np.ones(n*n)
    val = base.loglik(base.mat2vec(theta), g, pdata)
    print('val =', val)
    print('grd =')
    print(base.vec2mat(g, n))
    
    
if __name__ == '__main__':
    learn()
