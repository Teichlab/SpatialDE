import numpy as np
import pandas as pd

import SpatialDE as sde


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
    
    X = sample_info[['x', 'y']]

    # Convert data to log-scale
    dfm = np.log10(df + 1)

    # Perform Spatial DE test with default settings
    results = sde.run(X, dfm)

    # Save results and annotation in files for interactive plotting and interpretation
    sample_info.to_csv('MOB_sample_info.csv')
    results.to_csv('MOB_final_results.csv')

    return results 


if __name__ == '__main__':
    results = main()
