import sys
import numpy as np
import random
import itertools
import matplotlib.pylab as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as pltcm
import scipy.special
import scipy.stats
import collections

from cpp import diff


class MidpointNormalize(pltcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s)+1))


def mat2vec(mat):
    uidx = np.triu_indices(mat.shape[0], k=1)
    lidx = np.tril_indices(mat.shape[0], k=-1)
    upper = mat[uidx]
    lower = mat[lidx]
    diag = np.diag(mat)
    return np.hstack((diag, upper, lower))


def vec2mat(vec, n):
    uidx = np.triu_indices(n, k=1)
    lidx = np.tril_indices(n, k=-1)
    didx = np.diag_indices(n)
    ntri = uidx[0].shape[0]
    diag = vec[:n]
    upper = vec[n:n+ntri]
    lower = vec[n+ntri:]
    mat = np.zeros((n, n))
    mat[uidx] = upper
    mat[lidx] = lower
    mat[didx] = diag
    return mat


def permx_matrix(m, perm):
    if len(m.shape) == 1 or min(m.shape) <= 1:
        return np.asarray(m)[perm]
    elif len(m.shape) == 2:
        return np.asarray(m)[perm]
    else:
        raise Exception('Too many matrix dimensions.')


def permy_matrix(m, perm):
    if len(m.shape) != 2:
        raise Exception('Wrong number of dimensions')
    return np.asarray(m).T[perm].T


def plot_matrix(m, ax=None, xlabels=None, ylabels=None, text=None,
                vmin=None, vmax=None, vmid=None, notext=False,
                axes_fontsize=5, cell_fontsize=6, cmap='PiYG',
                grid=False, permx=None, permy=None, colorbar=False):
    nrows = m.shape[0]
    ncols = m.shape[1]
    if xlabels is None:
        xlabels = [str(i) for i in range(nrows)]
    if ylabels is None:
        ylabels = [str(i) for i in range(ncols)]
    if permx is not None:
        m = permx_matrix(m, permx)
        xlabels = [xlabels[x] for x in permx]
    if permy is not None:
        m = permy_matrix(m, permy)
        ylabels = [ylabels[x] for x in permy]
    nrows = m.shape[0]
    ncols = m.shape[1]
    if ax is None:
        ax = plt.gca()
    if vmin is None:
        vmin = np.min(m)
    if vmax is None:
        vmax = np.max(m)
    if vmid is None:
        vmid = 0.5 * (vmin + vmax)

    import numpy.ma
    maskdiag = np.diag(np.ones(m.shape[0]))
    mdiag = numpy.ma.masked_where(maskdiag==0, m)
    ax.matshow(mdiag, cmap='Greys', vmin=-6, vmax=0)
    moff = numpy.ma.masked_where(maskdiag==1, m)
    ax.matshow(moff, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)

    ax.set_yticks(range(nrows))
    ax.set_yticklabels(xlabels, fontsize=axes_fontsize)
    ax.set_xticks(range(ncols))
    ax.tick_params(length=0.1)
    lmax = max([len(ylabel) for ylabel in ylabels])
    ylabels = [yl + ' ' * (lmax - len(yl)) for yl in ylabels]
    ax.set_xticklabels(ylabels, fontsize=axes_fontsize, rotation=45, ha='left')
    xs, ys = np.meshgrid(range(ncols), range(nrows))
    if not notext:
        for x, y, w in zip(xs.flatten(), ys.flatten(), m.flatten()):
            if text == 'rows':
                ax.text(x, y, xlabels[y], va='center', ha='center',
                        fontsize=cell_fontsize, color='white')
            elif np.abs(w) > 1e-1:
                ax.text(x, y, '{0:.1f}'.format(w),
                        va='center', ha='center', fontsize=cell_fontsize)
    if grid:
        ax.set_xticks([x - 0.5 for x in range(ncols)], minor=True)
        ax.set_yticks([x - 0.5 for x in range(nrows)], minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    if colorbar:
        cbar = plt.gcf().colorbar(mat)
        cbar.ax.tick_params(labelsize=axes_fontsize)


def plot_result(util, wmat, ipmat, xlabels=None, iptext=None):
    _, ax = plt.subplots(1, 3, figsize=(20, 6),
                         sharey='row', gridspec_kw={'width_ratios': [1, 4, 4]})
    mutil = np.atleast_2d(util).T
    plot_matrix(mutil, ax=ax[0], xlabels=xlabels,
                vmin=np.min(mutil), vmax=np.max(mutil))
    ax[0].set_title('Marginals', fontsize=10)
    plot_matrix(wmat, ax=ax[1], xlabels=xlabels)
    ax[1].set_title('Weights', fontsize=10)
    if ipmat is not None:
        plot_matrix(ipmat, ax=ax[2], xlabels=xlabels, ylabels=xlabels,
                    text=iptext, vmin=-1, vmax=1, cmap='BrBG')
        ax[2].set_title('Interactions', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_distribution(ps, labels, legend=None):
    totalwidth = 0.5
    width = totalwidth / len(ps)
    cmap = get_colormap(len(ps))
    for i, p in enumerate(ps):
        if legend:
            label = legend[i]
        else:
            label = None
        xs = np.asarray(range(len(p))) + width*(i-0.5)
        plt.bar(xs, p, width=width, alpha=0.5, color=cmap(i), label=label)
        plt.gca().set_xticks(np.asarray(range(len(p))))
        plt.gca().set_xticklabels(labels, fontsize=8, rotation=90, ha='center')
    if legend:
        plt.legend()
    plt.show()


def get_colormap(n, name='jet'):
    cm = plt.get_cmap(name)
    cnorm = pltcolors.Normalize(vmin=0, vmax=n-1)
    map = pltcm.ScalarMappable(norm=cnorm, cmap=cm)
    return lambda i: map.to_rgba(i)


def truncate_colormap(name, minval=0.5, maxval=1.0, n=1000):
    cmap = plt.get_cmap(name)
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def cmap2hex(name):
    cmap = plt.get_cmap(name)
    return [pltcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]


def dot():
    sys.stdout.write('.')
    sys.stdout.flush()


def dot10(extra=''):
    sys.stdout.write(f'| {extra}\n')
    sys.stdout.flush()


def conditional_decorator(dec, flag):
    def decorate(fun):
        return dec(fun) if flag else fun
    return decorate


def marg_seq(ss, idxs):
    ssnew = []
    for s in ss:
        snew = [x for x in s if x in idxs]
        ssnew.append(tuple(snew))
    return ssnew


def KLdist(th1, th2, ndep, nsamples):
    s2, _ = diff.draw(th2, nsamples)
    s2 = marg_seq(s2, list(range(ndep)))
    seqs = []
    for s in powerset(range(th1.shape[0])):
        for p in itertools.permutations(s):
            seqs.append(p)

    c2 = collections.Counter(s2)
    p2 = np.array([(c2[s])/nsamples for s in seqs])

    logp1 = np.array([diff.loglik_seq(th1, seq)[0] for seq in seqs])
    lse = scipy.special.logsumexp(logp1)
    logp1 -= lse
    p1 = np.exp(logp1)

    return scipy.stats.entropy(p2, p1)


def order_data(data, fixed=None, extra=0):
    if fixed is None:
        fixed = []
    fixedidx = data.idx(fixed)
    rest = list(set(range(data.nitems)) - set(fixedidx))
    margs = [data.marginals[idx] for idx in rest]
    comb = zip(rest, margs)
    comb = sorted(comb, key=lambda x: x[1], reverse=True)
    rest, _ = zip(*comb)
    keep = fixedidx + list(rest[:extra])
    data = data.subset(keep)
    return data
