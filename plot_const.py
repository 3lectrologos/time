import numpy as np
import matplotlib.pylab as plt


w = np.linspace(0.01, 2, 100)
cs = [w]
ts = [0.5, 1, 2, 4]
for t in ts:
    c1 = np.exp(w*t) / w
    cs.append(c1)
    plt.plot(w, c1)
cs = np.array(cs).T
np.savetxt('/mnt/c/Users/el3ct/Desktop/timepaper/figures/c1.dat', cs)
plt.ylim((0, 20))
plt.show()

w = np.linspace(0.01, 2, 100)
cs = [w]
ts = [0.5, 1, 2, 4]
for t in ts:
    w = np.linspace(0.01, 2, 100)
    c2 = (2 - np.exp(-w*t)) / (w*w*np.exp(-w*t))
    cs.append(c2)
    plt.plot(w, c2)
cs = np.array(cs).T
np.savetxt('/mnt/c/Users/el3ct/Desktop/timepaper/figures/c2.dat', cs)
plt.ylim((0, 200))
plt.show()
