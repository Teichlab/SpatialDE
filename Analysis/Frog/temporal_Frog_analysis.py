import numpy as np
import pandas as pd

import SpatialDE as sde

def main():
    df = pd.read_csv('data/GSE65785_clutchApolyA_relative_TPM.csv', index_col=0)
    df = df[df.sum(1) >= 3]  # Filter practically unobserved genes

    # Get time points for each sample
    sample_info = pd.read_csv('data/sample_info.csv', index_col=0)

    X = sample_info[['hpf']]

    # Convert expression data to log scale, with genes in columns
    dfm = np.log10(df + 1).T

    # Perform Spatial DE test with default settings
    results = sde.run(X, dfm)

    # Save results and annotation in files for interactive plotting and interpretation
    sample_info.to_csv('Frog_sample_info.csv')
    results.to_csv('Frog_final_results.csv')

    return results 


if __name__ == '__main__':
    results = main()
