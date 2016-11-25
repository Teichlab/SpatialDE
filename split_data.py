import numpy as np
import pandas as pd

pattern = 'http://www.spatialtranscriptomicsresearch.org/wp-content/uploads/2016/07/Rep{}_MOB_count_matrix-1.tsv'

NUM_NODES = 10
replicates = [11, 12]

for replicate in replicates:
    df = pd.read_table(pattern.format(replicate), index_col=0)

    for grp, df_part in df.T.groupby(np.arange(df.shape[1]) // (df.shape[1] // NUM_NODES)):
        df_part.T.to_csv('data/Rep{}_MOB_{}.csv'.format(replicate, grp))
