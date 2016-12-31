import numpy as np
import pandas as pd

fgp = __import__('FaST-GP')


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords

df = pd.read_csv('data/Rep11_MOB_24.csv', index_col=0)
sample_info = get_coords(df.index)

# Run workflow
l = 25
K = fgp.SE_kernel(sample_info[['x']], l)
U, S = fgp.factor(K)
UT1 = fgp.get_UT1(U)

for g in df.columns:
    y = np.log10(df[g] + 1)
    n = y.shape[0]

    UTy = fgp.get_UTy(U, y)

    max_ll = 0.
    max_delta = np.nan
    for delta in np.logspace(base=np.e, start=-10, stop=10, num=10):
        cur_ll = fgp.LL(delta, UTy, UT1, S, n)
        if cur_ll > max_ll:
            max_ll = cur_ll
            max_delta = delta

    print(g, max_ll, max_delta)
