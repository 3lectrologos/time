import numpy as np
import sim
import base
import util
from datasets import Data
from cpp import diff


#np.random.seed(4)
size = 8
#theta = np.random.uniform(-1, 1, (size, size))
np.random.seed()
theta = np.array([[0, 3], [0, -3]])
ndep = theta.shape[0]
nrest = 6
tind = np.random.uniform(-3, 0, nrest)
trest = np.diag(tind)
theta = np.block([[theta, np.zeros((ndep, nrest))],
                  [np.zeros((nrest, ndep)), trest]])

n = theta.shape[0]
print('theta =')
print(theta)
print()

eth = np.exp(theta)
t1 = eth[0, 0]
t2 = eth[1, 1]
t12 = eth[0, 1]
t21 = eth[1, 0]
p12 = (t1 / (1 + t1 + t2)) * (t2*t12 / (1 + t2*t12))
p21 = (t2 / (1 + t1 + t2)) * (t1*t21 / (1 + t1*t21))
print('ratio (true) =', p12 / p21)
print('inv. ratio (true) =', p21 / p12)
print('prob (true) =', p12 / (p12 + p21))


data = sim.draw(theta, 10000)
p12 = len([d for d in data if d == [0, 1]])
p21 = len([d for d in data if d == [1, 0]])
print('p12, p21 =', p12, p21)
print('prob (sim) =', p12 / (p12 + p21))

s = [3, 0, 4]

tvec = util.mat2vec(theta)
#print('tvec =', tvec)
g = np.zeros_like(tvec)
data = Data.from_list([[s]], n)
#print(data)
base.loglik(tvec, g, base.get_pdata(data))
clograd = -util.vec2mat(g, n)
#print('grad (closed) =', clograd)

truegrad = diff.loglik_set_full(theta, s)[1]
#print('grad (full) =')
#print(truegrad)

NSAMPLES = 50

from timeit import default_timer
print()
start = default_timer()
grad = diff.loglik_set_unif(theta, s, NSAMPLES)
end = default_timer()
print(f'Time: {end-start}')
#print('grad =')
#print(grad)

print('DIF per element (unif) =', np.sum(np.abs(grad-clograd)) / (n*n))


print()
start = default_timer()
grad = diff.loglik_set(theta, s, NSAMPLES)
end = default_timer()
print(f'Time: {end-start}')
#print('grad =')
#print(grad)

print('DIF per element (new) =', np.sum(np.abs(grad-clograd)) / (n*n))
