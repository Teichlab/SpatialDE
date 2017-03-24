import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


def main():
    df = pd.read_csv('data/Rep11_MOB_0.csv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes
    
    # Get coordinates for each sample
    sample_info = get_coords(df.index)
    sample_info['total_counts'] = df.sum(1)
    sample_info = sample_info.query('total_counts > 10')  # Remove empty features
    df = df.loc[sample_info.index]
    
    X = sample_info[['x', 'y']]

    # Convert data to log-scale, and account for depth
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T

    # Perform Spatial DE test with default settings
    results = SpatialDE.run(X, res)

    # Save results and annotation in files for interactive plotting and interpretation
    sample_info.to_csv('MOB_sample_info.csv')
    results.to_csv('MOB_final_results.csv')

    return results 


if __name__ == '__main__':
    results = main()
