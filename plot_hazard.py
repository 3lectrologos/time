def run_hazard():
    #nreps = 10
    data = datasets.hazard()
    data = util.order_data(data, extra=100)
    res = learn.learn(data, reg=0.01, nsamples=50)
    #res = joblib.Parallel(n_jobs=nreps)(joblib.delayed(learn.learn)(data, show=False, niter=3000, step=0.2, reg=(1.0, 0.1),
    #                                                                exact=False, nsamples=10, init_theta='diag', verbose=True)
    #                                 for i in range(nreps))
    #with open('hazard.pcl', 'wb') as fout:
    #    pickle.dump(res, fout)


def plot_hazard():
    with open('hazard.pcl', 'rb') as fin:
        res = pickle.load(fin)
    std = np.std(res, axis=0)
    data = datasets.hazard()
    labels = []
    for label in data.labels:
        if label.endswith('(M)'):
            labels.append(label[:-3])
        else:
            labels.append(label)

    dif = np.amax(res, axis=0) - np.amin(res, axis=0)
    util.plot_matrix(dif,
                     xlabels=labels, ylabels=labels,
                     vmin=0, vmax=5, cmap='Greys', notext=False,
                     axes_fontsize=11, cell_fontsize=9)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig('hazard_dif.pdf', dpi=300)
    plt.gcf().clear()

    for i, r in enumerate(res):
        lik, _ = diff.loglik_data_full(r, data)
        print(f'{i}: {lik}')
        util.plot_matrix(r,
                         xlabels=labels, ylabels=labels,
                         vmin=-4, vmax=4, cmap='PuOr_r', notext=False,
                         axes_fontsize=11, cell_fontsize=9)
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(f'hazard_{i}.pdf', dpi=300)
        plt.gcf().clear()


def plot_big():
    plt.rcParams['axes.linewidth'] = 0.2
    with open(f'{DIR_NAME}/_150.pcl', 'rb') as fin:
        data, res = pickle.load(fin)

    nmax = 150
    idxs = range(nmax)

    margs = [data.marginals[idx] for idx in idxs]
    comb = zip(idxs, margs)
    comb = sorted(comb, key=lambda x: x[1], reverse=True)
    idxs, _ = zip(*comb)
    labels = [data.labels[x] for x in idxs]

    util.plot_matrix(res[0][np.ix_(idxs, idxs)],
                     xlabels=labels, ylabels=labels,
                     vmin=-4, vmax=4, cmap='PuOr_r', notext=True,
                     axes_fontsize=2)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig('gbm_big.pdf', dpi=300)
    plt.gcf().clear()
