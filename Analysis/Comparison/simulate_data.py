import click
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm

import SpatialDE


@click.command()
@click.argument('prefix')
@click.argument('n_samples', default=100)
@click.argument('n_genes', default=10000)
def make_data(prefix, n_samples, n_genes):
    X = np.random.uniform(size=(n_samples, 1), low=0, high=100)
    I = np.eye(n_samples)

    exp_tab = pd.DataFrame(index=range(n_samples))
    names = ['GP{}'.format(i) for i in range(n_genes)]
    true_values = pd.DataFrame(index=names, columns=['l', 'mu', 's2_t', 's2_e'])

    for g in tqdm(names):
        while True:
            l = np.exp(np.random.uniform(low=np.log(0.1), high=np.log(100)))
            mu = np.random.uniform(low=0., high=5.)
            s2_t = np.exp(np.random.uniform(low=-5., high=5.))
            s2_e = np.exp(np.random.uniform(low=-5., high=5.))

            K = SpatialDE.base.SE_kernel(X, l)

            mu1 = mu * np.ones((n_samples,))
            K_total = (s2_t * K + s2_e * I) 

            try:
                y = np.random.multivariate_normal(mu1, K_total)
            except np.linalg.linalg.LinAlgError:
                continue
            else:
                break

        exp_tab[g] = y
        true_values.loc[g, 'l'] = l
        true_values.loc[g, 'mu'] = mu
        true_values.loc[g, 's2_t'] = s2_t
        true_values.loc[g, 's2_e'] = s2_e

    pd.DataFrame(X, index=exp_tab.index).to_csv(prefix + '_X.csv')
    exp_tab.to_csv(prefix + '_expression.csv')
    true_values.to_csv(prefix + '_truth.csv')


if __name__ == '__main__':
    make_data()
