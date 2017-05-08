import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE


def main():
    df = pd.read_csv('exp_mat_43.csv', index_col=0)
    df.columns = df.columns.map(int)
    
    # Get coordinates for each sample
    sample_info = pd.read_csv('sample_info_43.csv', index_col=0)

    df = df[sample_info.index]
    
    X = sample_info[['x', 'y']]

    # Convert data to log-scale, and account for depth
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm, 'np.log(total_count)').T

    # Perform Spatial DE test with default settings
    results = SpatialDE.run(X, res)

    # Save results and annotation in files for interactive plotting and interpretation
    results.to_csv('final_results_43.csv')

    de_results = results[(results.qval < 0.05)].copy()
    ms_results = SpatialDE.model_search(X, res, de_results)

    ms_results.to_csv('MS_results_43.csv')

    return results 


if __name__ == '__main__':
    results = main()
 