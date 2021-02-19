import sys
import numpy as np
import random
import itertools
import functools
import csv
import matplotlib.pylab as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as pltcm
import scipy
from collections import defaultdict


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


def permute_rows(mat, seed=None, fixed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if fixed is None:
        fixed = []
    assert set(fixed) <= set(range(mat.shape[0]))
    rest = list(set(range(mat.shape[0]))-set(fixed))
    #ncols = mat.shape[1]
    for row in rest:
        np.random.shuffle(mat[row, :])
        #mat[row, :] = mat[row, np.random.permutation(ncols)]
    if seed is not None:
        np.random.seed()
        random.seed()


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
    mat = ax.matshow(m, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax,
                     norm=MidpointNormalize(midpoint=vmid), aspect='auto')
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
            else:
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


def perm_cmp(a, b, group):
    if a == b:
        return 0
    ova = [g for g in group if g in a]
    ovb = [g for g in group if g in b]
    if len(ova) == 0:
        return -1
    elif len(ovb) == 0:
        return 1
    pa = ova[0]
    pb = ovb[0]
    ia = group.index(pa)
    ib = group.index(pb)
    if ia < ib:
        return -1
    elif ia > ib:
        return 1
    else:
        return perm_cmp(list(set(a) - set([pa])),
                        list(set(b) - set([pb])),
                        group[::-1])


def set_axes_size(w, h, ax=None):
    if ax is None:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = w/(r-l)
    figh = h/(t-b)
    ax.figure.set_size_inches(figw, figh)


def perm_group(data, group, niter=10000, perm=True, mode='rep'):
    bs = defaultdict(lambda: [])
    for d in data:
        for g in group:
            if g in d:
                bs[g].append(d)
    fs = [(g, len(bs[g])) for g in group]
    if perm:
        fs = sorted(fs, key=lambda s: s[1], reverse=True)
    sgroup = next(zip(*fs))
    g2i = dict(zip(sgroup, range(len(sgroup))))
    mat = np.zeros((len(group), len(data)))
    datacopy = [list(set(group) & set(d))
                for d in data if len(set(group) & set(d)) != 0]
    datacopy = sorted(datacopy,
                      key=functools.cmp_to_key(
                          lambda a, b: perm_cmp(a, b, sgroup)))
    for i, d in enumerate(datacopy):
        if len(d) > 1:
            val = -1
        else:
            val = 1
        for x in d:
            mat[g2i[x], i] = val
    cov = (100.0 * len(datacopy)) / len(data)
    tab = data.matrix
    return mat, sgroup, cov


def ptest_string(pval):
    if pval < 0.001:
        return '{0:.2e}'.format(pval)
    else:
        return '{:.4f}'.format(pval)


def do_plot_perm_group(ax, data, group, title=None, showtitle=True, niter=10,
                       perm=True, mode='rep', axes_fontsize=8, title_fontsize=10,
                       **kwargs):
    mat, sgroup, cov = perm_group(data, group, niter, perm, mode)
    xlabels = [data.labels[g] for g in sgroup]
    ylabels = [''] * len(data)

    plot_matrix(mat, ax, vmin=-2.5, vmax=1.3, vmid=0,
                xlabels=xlabels, ylabels=ylabels,
                cmap='PuOr', notext=True, grid=False,
                axes_fontsize=axes_fontsize)
    plt.xticks([0, mat.shape[1]], labels=['0', str(mat.shape[1])], rotation=0, ha='center')
    ax.tick_params(axis='y', which='major', pad=8)


def plot_perm_group(data, group, dlen=None, **kwargs):
    fig, axes = plt.subplots(data.ntypes, 1)
    plt.rc('hatch', linewidth=4.5)
    if dlen is not None:
        plt.axvspan(len(data), dlen,
                    hatch='\\\\', facecolor='#999999', edgecolor='#eeeeee', linewidth=0)
        axes.set_xlim((0, dlen))
    height = 0.35*min(len(group), 15)
    ctop = 0.6
    bot = 0.01
    top = (height)/(ctop + height)
    plt.subplots_adjust(top=top, bottom=bot, left=0.13, right=0.98)
    set_axes_size(10, height)
    do_plot_perm_group(axes, data, group, **kwargs)


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


def dot10():
    sys.stdout.write('|\n')
    sys.stdout.flush()


def conditional_decorator(dec, flag):
    def decorate(fun):
        return dec(fun) if flag else fun
    return decorate
