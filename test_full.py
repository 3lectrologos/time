import numpy as np
import sim
import base
import util
import pydiff
from timeit import default_timer
from datasets import Data
from cpp import diff


#np.random.seed(4)
size = 20
theta = np.random.uniform(-2, 2, (size, size))
#np.random.seed()
#theta = np.array([[0, 3], [0, -3]])
#ndep = theta.shape[0]
#nrest = 6
#tind = np.random.uniform(-3, 0, nrest)
#trest = np.diag(tind)
#theta = np.block([[theta, np.zeros((ndep, nrest))],
#                  [np.zeros((nrest, ndep)), trest]])

#n = theta.shape[0]
#print('theta =')
#print(theta)
#print()

setsize = 8
s = np.random.choice(list(range(size)), setsize, replace=False)
#s = [3, 0, 1, 4, ]

#tvec = util.mat2vec(theta)
#g = np.zeros_like(tvec)
#data = Data.from_list([[s]], n)
#lik = -base.loglik(tvec, g, base.get_pdata(data))
#clograd = -util.vec2mat(g, n)
#print(f'lik (clo) =\t{lik:.8}')
#print()

truelik, truegrad = pydiff.loglik_set_new(theta, s)

funs = [pydiff.loglik_set_new, diff.loglik_set_full_old, diff.loglik_set_full, lambda t, x: (0, diff.loglik_set(t, x, 300))]
names = ['Python (new)', 'C++ (old)', 'C++ (new)', 'C++ (approx)']
for fun, name in zip(funs, names):
    start = default_timer()
    lik, grad = fun(theta, s)
    end = default_timer()
    ldif = np.abs(lik-truelik)
    gdif = np.sum(np.abs(grad-truegrad))
    print(name)
    print(f'lik =\t{ldif:.4}')
    print(f'gdif =\t{gdif:.4}')
    print(f'time =\t{1000*(end-start):.2f}ms') 
    print()
