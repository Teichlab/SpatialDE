import logging
from time import time

import numpy as np
import pandas as pd

import SpatialDE


def main():
    mll_results = pd.read_csv('../BreastCancer/BC_final_results.csv', index_col=0)

    sample_info = pd.read_csv('../BreastCancer/BC_sample_info.csv', index_col=0)
    X = sample_info[['x', 'y']]

    l_min, l_max = SpatialDE.base.get_l_limits(X)
    kernel_space = {
        'SE': np.logspace(np.log10(l_min), np.log10(l_max), 20),
        'const': 0
    }

    null_model_samples = 10000
    N = mll_results.loc[0, 'n']
    sim_null_exp_tab = SpatialDE.base.simulate_const_model(mll_results.sample(null_model_samples), N)

    logging.info('Performing DE test on null models'.format(null_model_samples))
    t0 = time()
    sim_results = SpatialDE.dyn_de(X, sim_null_exp_tab, kernel_space)
    sim_mll_results = SpatialDE.base.get_mll_results(sim_results)
    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    sim_mll_results.to_csv('BC_sim_results.csv')


if __name__ == '__main__':
    main()
