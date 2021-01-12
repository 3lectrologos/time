import csv
import random
import numpy as np
from collections import defaultdict


class DataError(Exception): pass


class Data:
    def __init__(self, data, labels=None):
        '''
        Args:
            data: A sequence of boolean NumPy arrays, each representing
                  one type.
            labels: A sequence of strings (same length as the ground set).
        '''
        self.ntypes = len(data)
        self.nitems = data[0].shape[0]
        for m in data:
            if m.shape[0] != self.nitems:
                raise DataError('Number of items mismatch')
        self._mats = [np.copy(m) for m in data]
        if labels is not None:
            assert len(set(labels)) == len(labels) == self.nitems
            self.labels = list(labels)
        else:
            self.labels = [str(x) for x in range(self.nitems)]
        self.label2id = dict(zip(self.labels, range(self.nitems)))

        if len(set(self.shortlabels)) != self.nitems:
            from collections import Counter
            cnt = Counter(self.shortlabels)
            for lab in self.shortlabels:
                if cnt[lab] > 1:
                    raise DataError(f'Short label \'{lab}\' is not unique')
        self.shortlabel2id = dict(zip(self.shortlabels, range(self.nitems)))

    @classmethod
    def from_list(cls, data, nitems, labels=None):
        '''
        Args:
            data: A sequence of lists of lists, each representing one type.
            nitems: The size of the ground set.
            labels: A sequence of strings (of length `nitems`).
        '''
        ntypes = len(data)
        mats = []
        for type in range(ntypes):
            nsamples = len(data[type])
            mat = np.zeros((nitems, nsamples), dtype='bool')
            for i, d in enumerate(data[type]):
                mat[d, i] = True
            mats.append(mat)
        return cls(mats, labels=labels)

    def __iter__(self):
        for m in self._mats:
            for j in range(m.shape[1]):
                yield list(np.where(m[:, j])[0])

    def __len__(self):
        return sum(m.shape[1] for m in self._mats)

    def __getitem__(self, j):
        for m in self._mats:
            tsamples = m.shape[1]
            if j < tsamples:
                return list(np.where(m[:, j])[0])
            else:
                j -= tsamples
        raise IndexError

    def __eq__(self, other):
        if self.ntypes != other.ntypes:
            return False
        if self.nitems != other.nitems:
            return False
        if self.labels != other.labels:
            return False
        for mself, mother in zip(self._mats, other._mats):
            if not np.array_equal(mself, mother):
                return False
        return True

    def copy(self):
        return Data(self._mats, self.labels)

    def __str__(self):
        s = f'{self.ntypes} type(s) | {self.nitems} genes x {len(self)} samples'
        return s

    def idx(self, label):
        try:
            if isinstance(label, list):
                return [self.label2id[lab] for lab in label]
            else:
                return self.label2id[label]
        except KeyError:
            if isinstance(label, list):
                return [self.shortlabel2id[lab] for lab in label]
            else:
                return self.shortlabel2id[label]

    def _label(self, idx):
        assert idx in range(self.nitems)
        return self.labels[idx]

    def label(self, idx):
        if isinstance(idx, list):
            return [self._label(x) for x in idx]
        else:
            return self._label(idx)

    @property
    def shortlabels(self):
        return [x[:25] for x in self.labels]

    def type(self, type):
        if type is None or (type == 0 and self.ntypes == 1):
            return self
        elif type in range(self.ntypes):
            return Data([self._mats[type]], labels=self.labels)
        else:
            raise IndexError(f'Unknown type {type}')

    @property
    def matrix(self):
        return np.hstack(self._mats)

    @property
    def marginals(self):
        return np.mean(self.matrix, axis=1)

    def batch(self, batch_size):
        batch_size = min(batch_size, len(self))
        batch_idxs = np.arange(len(self))
        np.random.shuffle(batch_idxs)
        batch_idxs = np.array_split(batch_idxs, len(self)/batch_size)
        for batch_idx in batch_idxs:
            # TODO: This listcomp is computationally inefficient.
            batch_list = [[self[idx] for idx in batch_idx]]
            yield Data.from_list(batch_list, self.nitems, labels=self.labels)

    def subset(self, idxs):
        idxs = list(idxs)
        assert set(idxs) <= set(range(self.nitems))
        newmats = []
        for type in range(self.ntypes):
            newmats.append(np.copy(self._mats[type][idxs, :]))
        newlabels = [self.labels[i] for i in idxs]
        return Data(newmats, newlabels)

    def permute_items(self, perm):
        assert sorted(list(perm)) == list(range(self.nitems))
        for type in range(self.ntypes):
            self._mats[type] = self._mats[type][perm, :]
        permlabels = [self.labels[i] for i in perm]
        self.labels = permlabels

    def bootstrap(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        newmats = []
        for type in range(self.ntypes):
            size = self._mats[type].shape[1]
            idxs = np.random.choice(range(size), size, replace=True)
            newmats.append(self._mats[type][:, idxs])
        if seed is not None:
            np.random.seed()
            random.seed()
        return Data(newmats, list(self.labels))

    def extend(self, data):
        '''
        Appends items (rows) at the end of the current data matrices
        for each type.

        Args:
            data: `Data` object with same number of types and number
                  of samples per type, but distinct labels.
        '''
        newmats = []
        if len(set(data.labels) & set(self.labels)) > 0:
            raise DataError('Overlapping labels')
        if data.ntypes != self.ntypes:
            raise DataError('Number of types mismatch')
        for type in range(self.ntypes):
            if data._mats[type].shape[1] != self._mats[type].shape[1]:
                raise DataError('Number of samples mismatch')
            newmats.append(
                np.vstack((self._mats[type], data._mats[type])))
        newlabels = self.labels + data.labels
        return Data(newmats, newlabels)

    def combine(self, genes, newlabel):
        '''
        Removes given genes (labels), and creates a new supergene with
        label `newlabel`, which is the union of the given genes.
        '''
        assert newlabel not in (set(self.labels)-set(genes))
        exgenes = list(set(genes)&set(self.labels))
        if exgenes == []:
            return self
        idxs = self.idx(exgenes)
        newmats = []
        for type in range(self.ntypes):
            new = np.zeros((1, self._mats[type].shape[1]))
            new = np.sum(self._mats[type][idxs, :], axis=0) >= 1
            new = np.atleast_2d(new)
            newmats.append(new)
        newdata = Data(newmats, [newlabel])
        sub = self.subset(list(set(range(self.nitems))-set(idxs)))
        sub = sub.extend(newdata)
        return sub

    # FIXME: use `util.permute_rows` here
    def rewire(self, seed=None, fixed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if fixed is None:
            fixed = []
        assert set(fixed) <= set(range(self.nitems))
        rest = list(set(range(self.nitems))-set(fixed))
        # for i, mat in enumerate(self._mats):
        #     self._mats[i][rest, :] = rewire.rewire(mat[rest, :])
        for mat in self._mats:
            for row in rest:
                np.random.shuffle(mat[row, :])
        if seed is not None:
            np.random.seed()
            random.seed()

    def perturb(self, pnoise, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        newmats = []
        for type in range(self.ntypes):
            tshape = self._mats[type].shape
            noise = np.random.binomial(1, pnoise, tshape[0]*tshape[1])
            noise = noise.reshape((tshape[0], tshape[1]))
            newmats.append(self._mats[type] ^ noise)
        if seed is not None:
            np.random.seed()
            random.seed()
        labels = list(self.labels)
        return Data(newmats, labels)        


    # TODO: May be better to pull all these class methods out of `Data`.
    @classmethod
    def _check_shape(cls, ds):
        for d in ds:
            if d.ntypes != ds[0].ntypes:
                raise DataError('Number of types mismatch')
            if d.nitems != ds[0].nitems:
                raise DataError('Number of items mismatch')
            if set(d.labels) != set(ds[0].labels):
                print(d.labels, ds[0].labels)
                raise DataError('Label mismatch')

    @classmethod
    def _check_types(cls, ds):
        for d in ds:
            if d.ntypes != ds[0].ntypes:
                raise DataError('Number of types mismatch')

    @classmethod
    def _check_labels(cls, ds):
        for d in ds:
            if d.labels != ds[0].labels:
                raise DataError('Label mismatch (need to be identical)')

    @classmethod
    def align_labels(cls, ds):
        cls._check_shape(ds)
        for d in ds:
            perm = [d.labels.index(x) for x in ds[0].labels]
            d.permute_items(perm)

    @classmethod
    def merge(cls, ds):
        cls._check_shape(ds)
        cls._check_labels(ds)
        merged_mats = []
        for type in range(ds[0].ntypes):
            mat = np.hstack([d._mats[type] for d in ds])
            merged_mats.append(mat)
        return cls(merged_mats, labels=ds[0].labels)

    @classmethod
    def fillin_template(cls, ds):
        cls._check_types(ds)
        ntypes = ds[0].ntypes
        freqs = [defaultdict(lambda: 0) for t in range(ntypes)]
        totals = [defaultdict(lambda: 0) for t in range(ntypes)]
        for data in ds:
            for type, mat in enumerate(data._mats):
                cnts = mat.sum(axis=1)
                for label, cnt in zip(data.labels, cnts):
                    freqs[type][label] += cnt
                    totals[type][label] += mat.shape[1]
        for type in range(ntypes):
            for label in freqs[type].keys():
                freqs[type][label] /= totals[type][label]
        newds = []
        for data in ds:
            # Sorting here makes sure that random seed works properly when
            # filling in later.
            missing = sorted(list(set(freqs[0].keys()) - set(data.labels)))
            if not missing:
                newds.append(data.copy())
                continue
            xmats = []
            for type in range(data.ntypes):
                nsamples = data._mats[type].shape[1]
                xmat = np.vstack(
                    [freqs[type][i]*np.ones(nsamples) for i in missing]
                )
                xmats.append(xmat)
            xdata = cls(xmats, labels=missing)
            newds.append(data.extend(xdata))
        return newds

    @classmethod
    def fillin(cls, ds, zeros=False, seed=None):
        ts = cls.fillin_template(ds)
        cls.align_labels(ts)
        return [t.draw_fillin(zeros, seed) for t in ts]

    @classmethod
    def fillin_and_merge_template(cls, ds):
        ts = cls.fillin_template(ds)
        cls.align_labels(ts)
        temp = cls.merge(ts)
        return temp

    def draw_fillin(self, zeros=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        res = self.copy()
        if zeros:
            for t in range(self.ntypes):
                res._mats[t][(res._mats[t]!=0)&(res._mats[t]!=1)] = 0
        else:
            unif = [np.random.uniform(size=m.shape) for m in self._mats]
            res._mats = [unif[t] < self._mats[t] for t in range(self.ntypes)]
        np.random.seed()
        random.seed()
        return res

    @classmethod
    def fillin_and_merge(cls, ds, zeros=False, seed=None):
        temp = cls.fillin_and_merge_template(ds)
        return temp.draw_fillin(zeros, seed)

    # Get samples for which all genes in `group` are observed (i.e., not
    # potentially filled-in). For a normal data instance, this returns the
    # full matrix of `type`.
    def obsmat(self, type, group):
        res = self._mats[type][group, :]
        zone = (res == 0) | (res == 1)
        valid = np.sum(zone, axis=0) == len(group)
        return np.array(res[:, valid], dtype='bool')

    @classmethod
    def intersect(cls, ds):
        cls._check_types(ds)
        labels_inter = set(ds[0].labels)
        for d in ds:
            labels_inter &= set(d.labels)
        newds = []
        for d in ds:
            idxs = [d.labels.index(lab) for lab in labels_inter]
            newd = d.subset(idxs)
            newds.append(newd)
        return newds

    @classmethod
    def intersect_and_merge(cls, ds):
        newds = Data.intersect(ds)
        Data.align_labels(newds)
        return Data.merge(newds)


# Comet parameters
MP = [0.5, 0.35, 0.15]
#MP = [0.4, 0.25, 0.2, 0.15]
#MP = [0.3, 0.2, 0.2, 0.15, 0.15]
GC = [0.67, 0.49, 0.29, 0.29, 0.2]
Q_NOISE = 0.002753


def comet_synthetic_single(gp, genes=50, patients=500, q_noise=None, seed=None):
    assert sum(MP) == 1.0
    if q_noise is None:
        q_noise = Q_NOISE
    # TODO: Create method for setting seeds
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    mat = np.zeros((genes, patients), dtype=bool)
    # Mutually-exclusive mutations
    p = np.random.choice(genes, len(MP), replace=False)
    npats = int(gp*patients)
    pidx = np.random.choice(patients, npats, replace=False)
    np.random.shuffle(pidx)
    nums = [0] + [int(x*npats) for x in np.cumsum(MP)]
    for k, prob in enumerate(MP):
        for j in range(nums[k], nums[k+1]):
            mat[p[k], pidx[j]] = True
    # Non-exclusive mutations
    candidates = list(set(range(genes)) - set(p))
    c = np.random.choice(candidates, len(GC), replace=False)
    for k, prob in enumerate(GC):
        cidx = np.random.choice(patients, int(GC[k]*patients), replace=False)
        for j in cidx:
            mat[c[k], j] = True
    # Noise
    noise = np.random.rand(genes, patients) < q_noise
    mat = np.logical_xor(mat, noise)
    # Create data object
    np.random.seed()
    random.seed()
    data = Data([mat])
    data.mutex = [list(p)]
    data.hyper = [list(c)]
    return data


GPS_LIST = [[0.7, 0.5],
            [0.7, 0.6, 0.5],
            [0.7, 0.6, 0.5, 0.4]]


def comet_synthetic_multi(t, k, total_genes=20000, patients=500,
                          q_noise=0.002753, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    assert t in [2, 3, 4]
    assert k in [3, 4, 5]
    gps = GPS_LIST[t-2]
    mpu = k * [1.0/k]
    mat = np.zeros((total_genes, patients), dtype=bool)
    # Mutually-exclusive mutations
    candidates = range(mat.shape[0])
    ps = []
    for ti in range(t):
        p = np.random.choice(candidates, len(mpu), replace=False)
        ps.append(p)
        pidx = np.random.choice(patients, int(gps[ti]*patients), replace=False)
        for j in pidx:
            i = np.random.choice(p, 1, mpu)
            mat[i, j] = True
        candidates = list(set(candidates) - set(p))
    # Non-exclusive mutations
    c = np.random.choice(candidates, len(GC), replace=False)
    for j in range(patients):
        for k, prob in enumerate(GC):
            if np.random.rand() < prob:
                mat[c[k], j] = True
    # Noise
    noise = np.random.rand(mat.shape[0], mat.shape[1]) < q_noise
    mat = np.logical_xor(mat, noise)
    # Remove genes that occur too rarely
    torem = []
    for i in range(mat.shape[0]):
        if np.sum(mat[i, :]) < 5:
            torem.append(i)
    mat = np.delete(mat, torem, 0)
    # Reindex p and c sets after deletions
    for ti in range(t):
        p = ps[ti]
        for i, pi in enumerate(p):
            shift = np.sum(np.asarray(torem) < pi)
            p[i] -= shift
    for i, ci in enumerate(c):
        shift = np.sum(np.asarray(torem) < ci)
        c[i] -= shift
    # Create data instance
    data = Data([mat])
    data.mutex = [list(pi) for pi in ps]
    data.hyper = c
    np.random.seed()
    random.seed()
    return data


def comet_2types(gp, k, genes=50, patients_per_type=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    gc_com = [0.7, 0.6, 0.5]
    gc_dif = 0.8
    mpu = k * [1.0/k]
    mats = []
    mats.append(np.zeros((genes, patients_per_type), dtype=bool))
    mats.append(np.zeros((genes, patients_per_type), dtype=bool))
    ps = []
    cs = []
    # Common mutually-exclusive mutation(s)
    candidates = range(genes)
    p = np.random.choice(candidates, len(mpu), replace=False)
    ps.append(p)
    for ti in [0, 1]:
        npats = int(gp*patients_per_type)
        pidx = np.random.choice(patients_per_type, npats, replace=False)
        np.random.shuffle(pidx)
        nums = [0] + [int(x*npats) for x in np.cumsum(mpu)]
        for k, prob in enumerate(mpu):
            for j in range(nums[k], nums[k+1]):
                mats[ti][p[k], pidx[j]] = True
    candidates = list(set(candidates) - set(p))
    # Different mutually-exclusive mutations (per type)
    for ti in [0, 1]:
        p = np.random.choice(candidates, len(mpu), replace=False)
        ps.append(p)
        npats = int(gp*patients_per_type)
        pidx = np.random.choice(patients_per_type, npats, replace=False)
        np.random.shuffle(pidx)
        nums = [0] + [int(x*npats) for x in np.cumsum(mpu)]
        for k, prob in enumerate(mpu):
            for j in range(nums[k], nums[k+1]):
                mats[ti][p[k], pidx[j]] = True
        candidates = list(set(candidates) - set(p))
    # Common non-exclusive mutations
    c = np.random.choice(candidates, len(gc_com), replace=False)
    cs.append(c)
    for ti in [0, 1]:
        for j in range(patients_per_type):
            for k, prob in enumerate(gc_com):
                if np.random.rand() < prob:
                    mats[ti][c[k], j] = True
    # Different non-exclusive mutations (per type)
    c = np.random.choice(candidates, 2, replace=False)
    cs.append(c)
    for ti in [0, 1]:
        for j in range(patients_per_type):
            if np.random.rand() < gc_dif:
                mats[ti][c[ti], j] = True
    mat = np.hstack((mats[0], mats[1]))
    # Noise
    noise = np.random.rand(mat.shape[0], mat.shape[1]) < Q_NOISE
    mat = np.logical_or(mat, noise)
    # Create data list
    nitems, nsamples = mat.shape
    data1, data2 = [], []
    for j in range(nsamples):
        d = np.asarray(range(nitems))
        d = list(d[mat[:, j]])
        if j < patients_per_type:
            data1.append(d)
        else:
            data2.append(d)
    np.random.seed()
    random.seed()
    data = Data([data1, data2], nitems)
    data.mutex = [list(pi) for pi in ps]
    data.hyper = [list(ci) for ci in cs]
    return data


def comet_3types(gp, k, genes=50, patients_per_type=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    gc_com = [0.7, 0.6, 0.5]
    gc_dif = 0.8
    mpu = k * [1.0/k]
    mats = []
    mats.append(np.zeros((genes, patients_per_type), dtype=bool))
    mats.append(np.zeros((genes, patients_per_type), dtype=bool))
    mats.append(np.zeros((genes, patients_per_type), dtype=bool))
    ps = []
    cs = []
    # Common mutually-exclusive mutations (to all types)
    candidates = range(genes)
    p = np.random.choice(candidates, len(mpu), replace=False)
    ps.append(p)
    for ti in [0, 1, 2]:
        npats = int(gp*patients_per_type)
        pidx = np.random.choice(patients_per_type, npats, replace=False)
        np.random.shuffle(pidx)
        nums = [0] + [int(x*npats) for x in np.cumsum(mpu)]
        for k, prob in enumerate(mpu):
            for j in range(nums[k], nums[k+1]):
                mats[ti][p[k], pidx[j]] = True
    candidates = list(set(candidates) - set(p))
    # Common mutually-exclusive mutations (to first two types)
    p = np.random.choice(candidates, len(mpu), replace=False)
    ps.append(p)
    for ti in [0, 1]:
        npats = int(gp*patients_per_type)
        pidx = np.random.choice(patients_per_type, npats, replace=False)
        np.random.shuffle(pidx)
        nums = [0] + [int(x*npats) for x in np.cumsum(mpu)]
        for k, prob in enumerate(mpu):
            for j in range(nums[k], nums[k+1]):
                mats[ti][p[k], pidx[j]] = True
    candidates = list(set(candidates) - set(p))
    # Different mutually-exclusive mutations (per type)
    for ti in [0, 1, 2]:
        p = np.random.choice(candidates, len(mpu), replace=False)
        ps.append(p)
        npats = int(gp*patients_per_type)
        pidx = np.random.choice(patients_per_type, npats, replace=False)
        np.random.shuffle(pidx)
        nums = [0] + [int(x*npats) for x in np.cumsum(mpu)]
        for k, prob in enumerate(mpu):
            for j in range(nums[k], nums[k+1]):
                mats[ti][p[k], pidx[j]] = True
        candidates = list(set(candidates) - set(p))
    # Common non-exclusive mutations
    c = np.random.choice(candidates, len(gc_com), replace=False)
    cs.append(c)
    for ti in [0, 1, 2]:
        for j in range(patients_per_type):
            for k, prob in enumerate(gc_com):
                if np.random.rand() < prob:
                    mats[ti][c[k], j] = True
    # Different non-exclusive mutations (per type)
    c = np.random.choice(candidates, 3, replace=False)
    cs.append(c)
    for ti in [0, 1, 2]:
        for j in range(patients_per_type):
            if np.random.rand() < gc_dif:
                mats[ti][c[ti], j] = True
    mat = np.hstack((mats[0], mats[1], mats[2]))
    # Noise
    noise = np.random.rand(mat.shape[0], mat.shape[1]) < Q_NOISE
    mat = np.logical_xor(mat, noise)
    # Create data as sets
    nitems, nsamples = mat.shape
    if False:
        data = []
        idxs = list(range(nsamples))
        np.random.shuffle(idxs)
        for j in idxs:
            d = np.asarray(range(nitems))
            d = list(d[mat[:, j]])
            data.append(d)
    else:
        data1, data2, data3 = [], [], []
        for j in range(nsamples):
            d = np.asarray(range(nitems))
            d = list(d[mat[:, j]])
            if j < patients_per_type:
                data1.append(d)
            elif j < 2*patients_per_type:
                data2.append(d)
            else:
                data3.append(d)
    np.random.seed()
    random.seed()
    data = Data.from_list([data1, data2, data3], nitems)
    data.mutex = [list(pi) for pi in ps]
    data.hyper = [list(ci) for ci in cs]
    return data


def comet_multi2(t, k, genes=100, patients=500,
                 q_noise=0.002753, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    assert t in [2, 3, 4]
    assert k in [3, 4, 5]
    gps = GPS_LIST[t-2]
    mpu = k * [1.0/k]
    mat = np.zeros((genes, patients), dtype=bool)
    # Mutually-exclusive mutations
    candidates = range(mat.shape[0])
    ps = []
    for ti in range(t):
        p = np.random.choice(candidates, len(mpu), replace=False)
        ps.append(p)
        pidx = np.random.choice(patients, int(gps[ti]*patients), replace=False)
        for j in pidx:
            i = np.random.choice(p, 1, mpu)
            mat[i, j] = True
        candidates = list(set(candidates) - set(p))
    # Non-exclusive mutations
    c = np.random.choice(candidates, len(gc), replace=False)
    for j in range(patients):
        for k, prob in enumerate(gc):
            if np.random.rand() < prob:
                mat[c[k], j] = True
    # Noise
    noise = np.random.rand(mat.shape[0], mat.shape[1]) < q_noise
    mat = np.logical_xor(mat, noise)
    # Create data instance
    data = Data([mat])
    data.mutex = [list(pi) for pi in ps]
    data.hyper = c
    np.random.seed()
    random.seed()
    return data


def two_panels():
    mul = 50
    d1 = [[0]]*2*mul + [[1]]*mul + [[2]]*mul + [[1, 2]]*mul + [[]]*(3*mul)
    data1 = Data.from_list([d1], 3, [str(x) for x in [0, 1, 2]])
    d2 = [[0]]*2*mul + [[1]]*mul + [[2]]*mul + [[1, 2]]*mul + [[]]*(3*mul)
    data2 = Data.from_list([d2], 3, [str(x) for x in [0, 1, 3]])
    return data1, data2


def overlap(save=True, seed=42):
    if seed is not None:
        print('seed =', seed)
        np.random.seed(seed)
        random.seed(seed)
    nitems = 15
    nsamples = 200
    nset = [150, 120, 90, 90, 90]
    mat = np.zeros((nitems, nsamples), dtype=bool)
    for i, novl in enumerate([0, 0, 0, 1, 2]):
        i1 = 3 * i
        i2 = 3 * i + 1
        i3 = 3 * i + 2
        js = np.random.choice(range(nsamples), nset[i]-3*novl, replace=False)
        idx12 = np.random.choice(js, novl, replace=False)
        js = list(set(js) - set(idx12))
        idx13 = np.random.choice(js, novl, replace=False)
        js = list(set(js) - set(idx13))
        idx23 = np.random.choice(js, novl, replace=False)
        js = list(set(js) - set(idx23))
        np.random.shuffle(js)
        idx = np.array_split(js, 3)
        idx1, idx2, idx3 = idx[0], idx[1], idx[2]
        print('novl =', novl, '--', len(idx1), len(idx2), len(idx3))
        for j in idx1:
            mat[i1, j] = True
        for j in idx2:
            mat[i2, j] = True
        for j in idx3:
            mat[i3, j] = True
        for j in idx12:
            mat[i1, j] = True
            mat[i2, j] = True
        for j in idx13:
            mat[i1, j] = True
            mat[i3, j] = True
        for j in idx23:
            mat[i2, j] = True
            mat[i3, j] = True
    # Remove empty sets
    torem = []
    for j in range(nsamples):
        if np.sum(mat[:, j]) == 0:
            torem.append(j)
    mat = np.delete(mat, torem, 1)
    nitems, nsamples = mat.shape
    # Create data as sets
    data = []
    for j in range(nsamples):
        d = np.asarray(range(nitems))
        d = list(d[mat[:, j]])
        data.append(d)
    labels = [str(x) for x in range(nitems)]
    np.random.seed()
    random.seed()
    ps = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    return data, nitems, 0, labels, (ps, [[]])


def two_large_ex(ngenes, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    nsamples = 500
    mat = np.zeros((ngenes, nsamples), dtype='bool')
    mat[0, :200] = 1
    mat[1, 170:400] = 1
    pnoise = 0.05
    for i in range(2, ngenes):
        mat[i, :] = np.random.binomial(1, pnoise, nsamples)
    return Data([mat])


def small(multiplier):
    data = [[]]*multiplier + [[0]]*multiplier +\
        [[1]]*multiplier + [[0, 2]]*multiplier +\
        [[1, 2]]*multiplier
    return Data(3, [data])


def synthetic4(ndata=100):
    data = []
    nfirst = 1
    nitems = 4
    for i in range(ndata):
        x1 = list(np.random.choice([0, 1], nfirst, replace=False,
                                   p=[0.6, 0.4]))
        x2 = list(np.random.choice([2, 3], 1, p=[0.2, 0.8]))
        x = x1 + x2
        random.shuffle(x)
        data.append(x)
    labels = [str(x) for x in range(nitems)]
    k = 2
    return data, nitems, k, labels


def synthetic6(ndata=100):
    data = []
    nfirst = 2
    nitems = 6
    for i in range(ndata):
        x1 = list(np.random.choice([0, 1, 2, 3], nfirst, replace=False,
                                   p=[0.4, 0.2, 0.1, 0.3]))
        x2 = list(np.random.choice([4, 5], 1, p=[0.7, 0.3]))
        x = x1 + x2
        random.shuffle(x)
        data.append(x)
    labels = [str(x) for x in range(nitems)]
    k = 3
    return data, nitems, k, labels


def ising4(ndata=1000):
    nitems = 4
    k = 2
    labels = [str(x) for x in range(nitems)]
    d1 = [[0, 2]] * (ndata / 2)
    d2 = [[1, 3]] * (ndata / 2)
    data = d1 + d2
    return data, nitems, k, labels


def toy_prop(prop):
    print('prop =', prop)
    nitems = 2
    data = [[]] * (1 * prop) +\
        [[0]] * (5 * prop) +\
        [[1]] * (2 * prop) +\
        [[0, 1]] * (1 * prop)
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy_over(over):
    print('over =', over)
    nitems = 2
    data = [[]] * (10 + over) +\
        [[0]] * (50 - over) +\
        [[1]] * (20 - over) +\
        [[0, 1]] * over
    np.random.shuffle(data)
    return Data.from_list([data], nitems)


def toy_none(none):
    print('none =', none)
    nitems = 2
    data = [[]] * none + [[0]] * 50 + [[1]] * 20 + [[0, 1]] * 10
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy_fill(fill):
    print('fill =', fill)
    nitems = 2
    data = [[]] * (30 - fill) +\
        [[0]] * (60 - 2 * fill) +\
        [[1]] * fill +\
        [[0, 1]] * (2 * fill)
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def fisher_problems_1():
    nitems = 3
    data = [[]] * 5 + [[0]] * 82 + [[1, 2]] * 7 + [[0, 1, 2]] * 3
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def fisher_problems_2():
    nitems = 3
    data = [[]] * 30 + [[0]] * 35 + [[1]] * 25 + [[0, 1]] * 5 + [[2]] * 1
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy3_rep_att():
    nitems = 3
    data = [[]] * 10 + [[0, 2]] * 50 + [[1]] * 20 + [[0, 1, 2]] * 10
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy3_att_act1():
    nitems = 3
    data = [[]] * 20 + [[0]] * 34 + [[1]] * 14 +\
        [[0, 1]] * 26 + [[0, 1, 2]] * 50
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy3_att_act2():
    nitems = 3
    data = [[2]] * 20 + [[0, 2]] * 34 + [[1, 2]] * 14 +\
        [[0, 1, 2]] * 26 + [[0, 1]] * 50
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy3_rep_act1():
    nitems = 3
    data = [[]] * 20 + [[0]] * 34 + [[1]] * 14 + [[0, 1]] * 26 +\
        [[0, 2]] * 25 + [[1, 2]] * 25
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy3_rep_act1_new():
    nitems = 3
    data = [[]] * 20 + [[0]] * 10 + [[1]] * 3 + [[0, 1]] * 10 +\
        [[0, 2]] * 30 + [[1, 2]] * 7 + [[2]] * 20
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy3_rep_act2():
    nitems = 3
    data = [[2]] * 20 + [[0, 2]] * 34 + [[1, 2]] * 14 +\
        [[0, 1, 2]] * 26 + [[0]] * 25 + [[1]] * 25
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy4_rep_att():
    nitems = 4
    data = [[]] * 10 + [[0]] * 30 + [[1]] * 20 + [[2, 3]] * 15 +\
        [[0, 1]] * 3 + [[1, 2, 3]] * 2 + [[0, 2, 3]] * 1
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy4_pairs():
    nitems = 6
    data = [[]] * 10 + [[0, 3]] * 30 + [[1, 4]] * 20 + [[2, 5]] * 10
    np.random.shuffle(data)
    print('len =', len(data))
    labels = [str(x) for x in range(nitems)]
    return data, nitems, 0, labels, ([[]], [[]])


def toy_over_triplets():
    nitems = 4
    data = [[]] * 10 + [[0]] * 50 + [[1]] * 50 +\
        [[2]] * 20 + [[3]] * 20 + [[2, 3]]*30
    np.random.shuffle(data)
    return Data.from_list([data], nitems)


def comet_real(name, use_types=False):
    fgenes = 'data/comet_' + name + '_genes.txt'
    fdata = 'data/comet_' + name + '.txt'
    if use_types:
        ftypes = 'data/comet_' + name + '_types.txt'
        fpats = 'data/comet_' + name + '_patient_types.txt'
        types = []
        with open(ftypes, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                types.append(row[0])
        type2idx = dict(zip(types, range(len(types))))
        pat2idx = {}
        with open(fpats, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                pat2idx[row[0]] = type2idx[row[1]]
    with open(fgenes, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        genes = []
        for row in reader:
            if row[0][0] == '#':
                continue
            genes.append(row[0])
    with open(fdata, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        genesets = []
        patient_ids = []
        for row in reader:
            patient_ids.append(row[0])
            g = [g for g in row[1:] if g in genes]
            genesets.append(list(set(g)))
        ngenes = len(genes)
        g2i = dict(zip(genes, range(ngenes)))
        i2g = dict(zip(range(ngenes), genes))
        if use_types:
            data = [[] for i in range(len(types))]
        else:
            data = []
        for i, geneset in enumerate(genesets):
            if len(geneset) == 0:
                print('Careful! Empty gene set in data.')
            dpoint = [g2i[g] for g in geneset]
            if use_types:
                try:
                    typeid = pat2idx[patient_ids[i]]
                except KeyError:
                    continue
                data[typeid].append(dpoint)
            else:
                data.append(dpoint)
        labels = [i2g[i] for i in range(ngenes)]
        return Data.from_list([data], ngenes, labels=labels)


def replace(data, orig, rep):
    cand = [x for x in enumerate(data.labels) if x[1].startswith(orig)]
    assert len(cand) == 1
    data.labels[cand[0][0]] = rep


def comet_aml():
    data = comet_real('aml')
    replace(data, 'ABL1,DYR', 'Tyr. Kinases')
    replace(data, 'ACVR2B,ADR', 'Ser/Thr. Kinases')
    replace(data, 'PTPN11,PTPRT', 'PTPs')
    replace(data, 'CSTF2T,DDX', 'Spliceosome')
    replace(data, 'SMC1A,SMC3', 'Cohesin')
    replace(data, 'MLL-ELL,MLL-MLLT', 'MLL-fusions')
    replace(data, 'ARID4B,ASXL2,', 'MODs')
    replace(data, 'GATA2,CBFB', 'Myeloid TFs')
    return Data([data.matrix], labels=data.labels)


def comet(type):
    if type == 'aml':
        return comet_aml()
    else:
        return comet_real(type)
