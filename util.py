import sys
import numpy as np
import random
import itertools
import functools
import csv
import sklearn
import sklearn.cluster as skc
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


def mat2svg_count(mat, idxs):
    ind = np.sum(mat[idxs, :], axis=0) == len(idxs)
    msum = np.sum(mat, axis=0) == len(idxs)
    return np.sum(ind & msum)


def svg_comp(r1, r2):
    if r1 == r2:
        return 0
    s1, s2 = sorted(r1, reverse=True), sorted(r2, reverse=True)
    n = min(len(s1), len(s2))
    for e1, e2 in zip(s1[:n], s2[:n]):
        if e1 != e2:
            return e2 - e1
    return len(s1) - len(s2)


def svg_comp_miss(r1, r2):
    if r1 == r2:
        return 0
    if len(r1) != len(r2):
        return len(r2) - len(r1)
    flex = lambda r: ''.join([str(x) for x in r])
    if flex(r1) < flex(r2):
        return -1
    else:
        return 1


def svg_reduce(res):
    newres = [[[] for j in range(len(res[0]))] for i in range(len(res))]
    for j in range(len(res[0])):
        for k in range(len(res[0][0])):
            safe = True
            for i in range(len(res)):
                if res[i][j][k][1] != 0:
                    safe = False
                    break
            if not safe:
                for i in range(len(res)):
                    newres[i][j].append(res[i][j][k])
    return newres


def mat2svg(mat, sel, n, miss=False):
    sel2mat = dict(zip(sel, range(mat.shape[0])))
    res = [[] for i in range(n)]
    start = 0
    subs = []
    if not miss:
        for i in range(n):
            spow = sorted(powerset(range(i+1, n)), key=functools.cmp_to_key(svg_comp))
            subs += list([i] + list(x) for x in spow)
        subs.append([])
    else:
        subs = sorted(powerset(range(n)), key=functools.cmp_to_key(svg_comp_miss))

    for idxs in subs:
        if not set(idxs) <= set(sel):
            cnt = 0
        else:
            cnt = mat2svg_count(mat, [sel2mat[idx] for idx in idxs])
        for idx in range(n):
            if idx in idxs:
                if len(idxs) == 1:
                    col = 1
                else:
                    col = 2
            else:
                col = 0
            res[idx].append([start, cnt, col])
        start += cnt

    final = []
    for i in range(n):
        if i in sel:
            res[i].append([start, 0, 3])
            final.append(res[i])
        else:
            inactive = [[0, 0, x] for _, _, x in res[i]]
            inactive.append([0, mat.shape[1], 3])
            final.append(inactive)
    return final


def matcounts(mat):
    m = np.vstack((np.sum(mat[:-1, :], axis=0) >= 1,
                   mat[-1, :]))
    cb = np.sum(np.sum(m, axis=0) == 2)
    cx = np.sum(m[0, :]) - cb
    cy = np.sum(m[1, :]) - cb
    cn = mat.shape[1] - cx - cy - cb
    return cn, cx, cy, cb


def mat2svg_rob(mat, rob, sel, n):
    res = [[] for i in range(n)]
    group = range(mat.shape[0])
    types = [1, 5, 2, 4, 6, 0, 3]
    for i, g in enumerate(group):
        rest = list(set(group) - set([g]))
        cn, cx, cy, cb = matcounts(mat[rest + [g], :])
        r = rob[i][2]
        widths = [cy-r, r, cb, cx-r, r, cn, 0]
        starts = list(np.cumsum([0] + widths))[:-1]
        # This is for overlap visualization
        widths[0] = cy
        res[sel[i]] = [list(x) for x in zip(starts, widths, types)]
    for i in (set(range(n)) - set(sel)):
        widths = [0]*6 + [mat.shape[1]]
        starts = [0]*7
        res[i] = [list(x) for x in zip(starts, widths, types)]
    return res


def precision_recall(true, pred):
    if len(true) == 1 and type(true[0]) == list:
        true = true[0]
    true_set = set(true)
    pred_union = set()
    pred_union_two_or_more = set()
    for pr in pred:
        pred_union |= set(pr)
        tp = len(true_set & set(pr))
        if tp >= 2:
            pred_union_two_or_more |= set(pr)
    truepos = len(true_set & pred_union_two_or_more)
    falsepos = len(pred_union) - truepos
    falseneg = len(true_set - pred_union_two_or_more)
    if truepos + falsepos == 0:
        precision = 1.0
    else:
        precision = truepos / (truepos + falsepos)
    recall = truepos / (truepos + falseneg)
    return precision, recall


def fmeasure(true, pred):
    if len(true) == 1 and type(true[0]) == list:
        true = true[0]
    precision, recall = precision_recall(true, pred)
    if precision*recall == 0:
        fmeasure = 0
    else:
        fmeasure = 2.0 / (1.0 / recall + 1.0 / precision)
    return fmeasure


def avgprc_aux(fprc, true, pred, plot=False):
    precisions, recalls = [], []
    for p in pred:
        precision, recall = fprc(true, p)
        precisions.append(precision)
        recalls.append(recall)
    # Add values corresponding to empty prediction
    precisions.append(1)
    recalls.append(0)
    # Reverse to have recall go from 0 -> 1
    precisions = precisions[::-1]
    recalls = recalls[::-1]
    print('RECALLS:', recalls)
    avg_precision = 0
    for i in range(1, len(precisions)):
        recall_diff = recalls[i]-recalls[i-1]
        assert recall_diff >= 0
        avg_precision += precisions[i]*recall_diff
    if plot:
        plt.plot(recalls, precisions, 'bo')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()
    return avg_precision


def avgprc(true, pred, plot=False):
    return avgprc_aux(precision_recall, true, pred, plot)


def get_pairs(res):
    pairs = set()
    for s in res:
        for pair in itertools.combinations(s, 2):
            pairs.add(frozenset(pair))
    return pairs


def precision_recall_multi(true, pred):
    true_pairs = get_pairs(true)
    pred_pairs = get_pairs(pred)
    truepos = len(true_pairs & pred_pairs)
    falsepos = len(pred_pairs - true_pairs)
    falseneg = len(true_pairs - pred_pairs)
    if truepos + falsepos == 0:
        precision = 1.0
    else:
        precision = truepos / (truepos + falsepos)
    recall = truepos / (truepos + falseneg)
    return precision, recall


def avgprc_multi(true, pred, plot=False):
    return avgprc_aux(precision_recall_multi, true, pred, plot)


def adjusted_rand_index(true, res):
    allres = [x for s in res for x in s]
    alltrue = [x for s in true for x in s]
    total = list(set(allres) | set(alltrue))
    ntotal = len(total)
    t2i = dict(zip(total, range(ntotal)))
    cres = np.zeros(ntotal, dtype='int')
    ctrue = np.zeros(ntotal, dtype='int')
    for i, s in enumerate(true):
        for x in s:
            ctrue[t2i[x]] = i
    union = set(alltrue) | set(allres)
    kextra = 1000
    for x in set(total) - set(alltrue):
        ctrue[t2i[x]] = kextra
        kextra += 1
    for i, s in enumerate(res):
        for x in s:
            cres[t2i[x]] = i
    kextra = 2000
    for x in set(total) - set(allres):
        cres[t2i[x]] = kextra
        kextra += 1
    return sklearn.metrics.adjusted_rand_score(ctrue, cres)


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
    sys.stdout.write('|')
    sys.stdout.flush()


def conditional_decorator(dec, flag):
    def decorate(fun):
        return dec(fun) if flag else fun
    return decorate
