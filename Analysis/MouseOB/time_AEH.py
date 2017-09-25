from time import time

import numpy as np
import pandas as pd

import NaiveDE
from GPy import kern
import GPclust


def main():
    sample_info = pd.read_csv('MOB_sample_info.csv', index_col=0)

    df = pd.read_csv('data/Rep11_MOB_0.csv', index_col=0)
    df = df.loc[sample_info.index]
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes

    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T

    X = sample_info[['x', 'y']].values

    times = pd.DataFrame(columns=['N', 'time'])
    Ns =  [50, 100, 200, 300, 500, 750, 1000, 2000]

    j = 0
    for N in Ns:
        for i in range(5):

            Y = res.sample(N, axis=1).values.T

            t0 = time()

            m = GPclust.MOHGP(
                                X=X,
                                Y=Y,
                                kernF=kern.RBF(2) + kern.Bias(2),
                                kernY=kern.RBF(1) + kern.White(1),
                                K=5,
                                prior_Z='DP'
                            )

            m.hyperparam_opt_args['messages'] = False
            m.optimize(step_length=0.1, verbose=False, maxiter=2000)

            times.loc[j] = [N, time() - t0]
            print(times.loc[j])
            j += 1

    times.to_csv('AEH_times.csv')


if __name__ == '__main__':
    results = main()
