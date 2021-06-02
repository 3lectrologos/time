import numpy as np
import matplotlib.pylab as plt


a = 4
p0 = 1 / (1 + np.exp(0) + np.exp(-a))
p1 = p0*np.exp(0) / (1 + np.exp(0))
p2 = p0*np.exp(-a) / (1 + np.exp(0))

print(p0, p1, p2)

start = 0.5
end = (1/p0)-1 - 1e-3
mid = 1
t1 = np.hstack((np.linspace(start, mid, 500), np.linspace(mid, end, 1000)))
#t1 = np.geomspace(start, end, 1000)
t2 = (1/p0) - 1 - t1
t21 = (1-p0-p2-p0*t1) / (p2*t1)
#t12 = (1-p0-p1-p2-p2*t1*t21) / (p1*t2)
t12 = (p0*t1-p1) / (p1*t2)

np.savetxt('/mnt/c/Users/el3ct/Desktop/timepaper/figures/example_params_1.dat',
           np.vstack((np.log(t1), np.log(t2))).T, header='x\ty')
np.savetxt('/mnt/c/Users/el3ct/Desktop/timepaper/figures/example_params_2.dat',
           np.vstack((np.log(t12), np.log(t21))).T, header='x\ty')

istart = 55
iend = -600

plt.plot(np.log(t1), np.log(t2), '-', linewidth=2)
plt.plot(0, -a, 'ko', markersize=10)
plt.plot(np.log(t1)[istart], np.log(t2)[istart], 'kx', markersize=10)
plt.plot(np.log(t1)[iend], np.log(t2)[iend], 'kx', markersize=10)
plt.xlim((-0.58, 0.01))
plt.ylim((-4.5, -0.75))
plt.show()

plt.plot(np.log(t12), np.log(t21), '-', linewidth=2)
plt.plot(a, 0, 'ko', markersize=10)
plt.plot(np.log(t12)[istart], np.log(t21)[istart], 'kx', markersize=10)
plt.plot(np.log(t12)[iend], np.log(t21)[iend], 'kx', markersize=10)
plt.xlim((-1.5, 4.5))
plt.ylim((-1.5, 4.5))
plt.show()
