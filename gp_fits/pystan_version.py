# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from gp_fits.fit_gps import get_coords

import pystan

# %%
import pickle

gpr = pickle.load(open('gp_fits/gp_regression.pkl', 'rb'))

# %%
df = pd.read_csv('data/Rep11_MOB_24.csv', index_col=0)
sample_info = get_coords(df.index)

# %%
y = np.log10(df['Clip1'] + 1)
plt.scatter(sample_info.x, sample_info.y, c=y)
plt.colorbar()

# %%
data = {
    'N': sample_info.shape[0],
    'x': sample_info.as_matrix(),
    'y': y
}
gpr.optimizing(data=data)

# %%
results = []
for g in df.columns:
    y = np.log10(df[g] + 1)
    data = {
        'N': sample_info.shape[0],
        'x': sample_info.as_matrix(),
        'y': y
    }
    o = gpr.optimizing(data=data)
    o['gene'] = g
    results.append(o)

results = pd.DataFrame(results)

# %%
results

# %%
plt.style.use('bmh')
results['tot_var'] =  results[['s2_bias', 's2_model', 's2_se']].sum(1)
plt.scatter(results.s2_bias / results.tot_var * 100,
            results.s2_se / results.tot_var * 100, c='k', s=50)

plt.xlabel('% Bias variance')
plt.ylabel('% Spatial variance')

# %%
# plt.xscale('log')
plt.scatter(results.l,
            results.s2_se / results.tot_var * 100, c='k', s=50)
# plt.xlim(1e-2, 1e2)

plt.xlabel('Lengthscale')
plt.ylabel('% Spatial variance')

# %%
results.query('4. < l < 10.')

# %%
y = np.log10(df['Enah'] + 1)
plt.scatter(sample_info.x, sample_info.y, c=y, s=50, cmap=cm.magma)
plt.colorbar()
