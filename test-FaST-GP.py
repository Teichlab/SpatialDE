import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

fgp = __import__('FaST-GP')


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords

@profile
def main():
    df = pd.read_csv('data/Rep12_MOB_3.csv', index_col=0)
    sample_info = get_coords(df.index)

    # Run workflow
    X = sample_info[['x', 'y']]
    dfm = np.log10(df + 1)
    l = 10
    results = fgp.dyn_de(X, dfm, lengthscale=l, num=64)

    plt.scatter(results['max_delta'], results['max_ll'], c='k')
    plt.xscale('log')
    plt.xlim(np.exp(-11), np.exp(11))
    plt.xlabel('$\delta$')
    plt.ylabel('Maximum Log Likelihood')
    plt.title('lengthscale: {}'.format(l))
    plt.savefig('fastgp-fits.png', bbox_inches='tight')
    
    print(results.sort_values('max_delta').head(20))

if __name__ == '__main__':
    main()
