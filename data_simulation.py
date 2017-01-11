import numpy as np
import pandas as pd
from tqdm import tqdm

fgp = __import__('FaST-GP')

def make_ls_data(lengthscale, n_obs, n_sim):
    X = np.random.uniform(size=(n_obs, 1), low=0, high=100)
    K = fgp.SE_kernel(X, lengthscale)
    I = np.eye(n_obs)

    exp_tab = pd.DataFrame(index=range(n_obs))
    names = ['GP{}'.format(i) for i in range(n_sim)]
    true_values = pd.DataFrame(index=names, columns=['mu', 's2_t', 's2_e'])

    for g in names:
        mu = np.random.uniform(low=0., high=5.)
        s2_t = np.exp(np.random.uniform(low=-5., high=5.))
        s2_e = np.exp(np.random.uniform(low=-5., high=5.))

        y = np.random.multivariate_normal(mu * np.ones((n_obs,)), (s2_t * K + s2_e * I))
    
        exp_tab[g] = y
        true_values.loc[g, 'mu'] = mu
        true_values.loc[g, 's2_t'] = s2_t
        true_values.loc[g, 's2_e'] = s2_e

    return X, exp_tab, true_values


def make_multi_ls_data(l_min=1, l_max=100, n_obs=500, n_sim=500):
    X = np.random.uniform(size=(n_obs, 1), low=0, high=100)
    I = np.eye(n_obs)

    exp_tab = pd.DataFrame(index=range(n_obs))
    names = ['GP{}'.format(i) for i in range(n_sim)]
    true_values = pd.DataFrame(index=names, columns=['l', 'mu', 's2_t', 's2_e'])

    for g in tqdm(names):
        l = np.exp(np.random.uniform(low=np.log(l_min), high=np.log(l_max)))
        mu = np.random.uniform(low=0., high=5.)
        s2_t = np.exp(np.random.uniform(low=-5., high=5.))
        s2_e = np.exp(np.random.uniform(low=-5., high=5.))

        K = fgp.SE_kernel(X, l)

        y = np.random.multivariate_normal(mu * np.ones((n_obs,)), (s2_t * K + s2_e * I))

        exp_tab[g] = y
        true_values.loc[g, 'l'] = l
        true_values.loc[g, 'mu'] = mu
        true_values.loc[g, 's2_t'] = s2_t
        true_values.loc[g, 's2_e'] = s2_e

    return X, exp_tab, true_values
