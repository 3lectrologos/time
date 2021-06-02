import datasets


data = datasets.comet('gbm')
res = []
for label in data.labels:
    lst = label.split(',')
    for lab in lst:
        if lab.endswith('(D)') or lab.endswith('(A)'):
            res.append(lab[:-3])
        else:
            res.append(lab)
res = list(set(res))
print(f'Total: {len(res)}')
s = '\n'.join(res)
with open('gbm_genes.txt', 'w') as fout:
    fout.write(s)
