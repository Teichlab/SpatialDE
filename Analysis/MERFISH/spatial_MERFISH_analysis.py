import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE


def main():
    df = pd.read_csv('data/rep6/middle_exp_mat.csv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes

    # Get coordinates for each sample
    sample_info = pd.read_csv('data/rep6/middle_sample_info.csv', index_col=0)
    df = df.loc[sample_info.index]

    X = sample_info[['abs_X', 'abs_Y']]

    # Convert data to log-scale, and account for depth
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(cytoplasmArea)').T

    # Perform Spatial DE test with default settings
    results = SpatialDE.run(X, res)

    # Assign pi_0 = 1 in multiple testing correction
    results['qval'] = SpatialDE.util.qvalue(results['pval'], pi0=1.0)

    # Save results and annotation in files for interactive plotting and interpretation
    sample_info.to_csv('middle_sample_info.csv')
    results.to_csv('middle_final_results.csv')

    de_results = results[(results.qval < 0.05)].copy()
    ms_results = SpatialDE.model_search(X, res, de_results)

    ms_results.to_csv('middle_MS_results.csv')

    return results


if __name__ == '__main__':
    results = main()
