import os
import numpy as np
import pandas as pd
import csv
from .core import Data


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TCGA_DIR = os.path.join(BASE_DIR, 'tcga')
HAZARD_DIR = os.path.join(BASE_DIR, 'hazard')
COMET_DIR = os.path.join(BASE_DIR, 'comet')
MISC_DIR = os.path.join(BASE_DIR, 'misc')


combos = [
    (['CDKN2A(D)', 'CDKN2B(D)'], 'CDKN2A/B(D)'),
    (['KDR(A)', 'KIT(A)', 'PDGFRA(A)'], 'PDGFRA(A)'),
    (['CYSLTR2(D)', 'RB1(D)'], 'RB1(D)'),
    (['CDKN2C(D)', 'FAF1(D)'], 'CDKN2C(D)'),
    (['HIST2H3D(A)', 'NOTCH2(A)'], 'NOTCH2(A)'),
    (['FGF19(A)', 'FGF3(A)', 'FGF4(A)', 'CCND1(A)'], 'CCND1(A)'),
    (['PIK3CA(A)', 'SOX2(A)'], 'PIK3CA(A)')
]

coad_combos = [
    (['APC', 'FBXW7', 'MUC4'], 'L1'),
    (['ACVR2A', 'ELF3', 'PIK3CA', 'SOX9', 'TP53'], 'L2'),
    (['BRAF', 'KRAS', 'NRAS'], 'L3'),
    (['SMAD2', 'SMAD4'], 'L4')
]


def combine_genes(data, comb):
    newdata = data.copy()
    for genes, supergene in comb:
        newdata = newdata.combine(genes, supergene)
    return newdata


def get_mut_cna(type, mutonly=False):
    fmut = f'{TCGA_DIR}/tcga_{type}_mut.txt'
    mut = pd.read_csv(fmut, sep='\t', skiprows=1)
    genes_mut = mut.iloc[:, 0].to_list()
    ids_mut = list(mut.columns[1:])
    mat_mut = mut.iloc[:, 1:].to_numpy()
    mat_mut = mat_mut != 'WT'

    if not mutonly:
        fcna = f'{TCGA_DIR}/tcga_{type}_cna.txt'
        cna = pd.read_csv(fcna, sep='\t', skiprows=1)
        genes_cna = cna.iloc[:, 0].to_list()
        ids_cna = list(cna.columns[1:])
        mat_cna = cna.iloc[:, 1:].to_numpy(dtype='double')
        if np.sum(np.isnan(mat_cna)) != 0:
            assert False
        assert genes_mut == genes_cna
        assert ids_mut == ids_cna
        mat_amp = (mat_cna > 1)
        mat_del = (mat_cna < -1)
        genes_amp = [g + '(A)' for g in genes_cna]
        genes_del = [g + '(D)' for g in genes_cna]

        mat = np.vstack((mat_mut, mat_amp, mat_del))
        genes = genes_mut + genes_amp + genes_del
    else:
        mat = mat_mut
        genes = genes_mut
    data = Data([mat], labels=genes)
    data = combine_genes(data, combos)
    return data


def get_alt(type):
    falt = f'{TCGA_DIR}/tcga_{type}_matrix.txt'
    alt = pd.read_csv(falt, sep='\t')
    mat = alt.iloc[:, 2:].to_numpy().T
    genes = list(alt.columns[2:])
    data = Data([mat], labels=genes)
    # XXX: temp
    data = combine_genes(data, coad_combos)
    #
    return data


def tcga(type, alt=False, mutonly=False):
    if alt:
        return get_alt(type)
    else:
        return get_mut_cna(type, mutonly)


def bekka():
    fmut = f'{MISC_DIR}/colorectal49.txt'
    mut = pd.read_csv(fmut, sep='\t')
    genes = list(mut.columns)
    mat = mut.to_numpy().T
    return Data([mat], labels=genes)


def hazard():
    fmut = f'{HAZARD_DIR}/gbm.csv'
    df = pd.read_csv(fmut)
    mat = df.to_numpy().T
    labels = [col for col in df.columns]
    return Data([mat], labels=labels)


def comet_real(name, use_types=False):
    fgenes = f'{COMET_DIR}/comet_' + name + '_genes.txt'
    fdata = f'{COMET_DIR}/comet_' + name + '.txt'
    if use_types:
        ftypes = f'{COMET_DIR}/comet_' + name + '_types.txt'
        fpats = f'{COMET_DIR}/comet_' + name + '_patient_types.txt'
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


def lambda_struct():
    lst = [[]]*1 + [[0]]*2 + [[0, 1]]*4 + [[0, 2]]*3
    data = Data.from_list([lst], nitems=3)
    return data


def v_struct():
    lst = [[]]*20 + [[0]]*10 + [[1]]*10
    lst += [[0, 1]]*5 + [[0, 2]]*15 + [[1, 2]]*15 + [[0, 1, 2]]*3
    data = Data.from_list([lst], nitems=3)
    return data
