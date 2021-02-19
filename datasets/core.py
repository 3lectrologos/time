import csv
import random
import numpy as np
import pandas as pd
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
