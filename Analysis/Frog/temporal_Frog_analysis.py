import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE

def main():
    # Get time points for each sample
    sample_info = pd.read_csv('Frog_sample_info.csv', index_col=0)

    # Load expression
    df = pd.read_csv('data/GSE65785_clutchApolyA_relative_TPM.csv', index_col=0)
    df = df[sample_info.index]
    df = df[df.sum(1) >= 3]  # Filter practically unobserved genes

    X = sample_info[['hpf']]

    # Convert expression data to log scale, with genes in columns
    dfm = NaiveDE.stabilize(df)
    res = NaiveDE.regress_out(sample_info, dfm, 'np.log(ERCC) + np.log(num_genes)', rcond=1e-4).T

    # Perform Spatial DE test with default settings
    results = SpatialDE.run(X, res)

    # Save results and annotation in files for interactive plotting and interpretation
    results.to_csv('Frog_final_results.csv')

    return results 


if __name__ == '__main__':
    results = main()
