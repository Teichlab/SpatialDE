
SpatialDE
=========

SpatialDE is a method to identify genes which signfificantly depend on
spatial coordinates in non-linear and non-parametric ways. The intended
applications are spatially resolved RNA-sequencing from e.g. Spatial
Transcriptomics, or *in situ* gene expression measurements from e.g.
SeqFISH or MERFISH.

This repository contains both the implementations of our method, as well
as case studies in applying it.

The key features of our method are

-  Unsupervised - No need to define spatial regions
-  Non-parametric and non-linear expression patterns
-  Extremely fast - Transcriptome wide tests takes only a few minutes on
   normal computers

The primary implementation is as a Python3 package, and can be installed
from the command line by

::

    $ pip install spatialde

Below follows a typical usage example in interactive form.

.. code:: ipython3

    %pylab inline
    import pandas as pd
    
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    
    import NaiveDE
    import SpatialDE


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: ipython3

    counts = pd.read_csv('Analysis/MouseOB/data/Rep11_MOB_0.csv', index_col=0)
    counts = counts.T[counts.sum(0) >= 3].T  # Filter practically unobserved genes
    
    counts.iloc[:5, :5]




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Nrf1</th>
          <th>Zbtb5</th>
          <th>Ccnl1</th>
          <th>Lrrfip1</th>
          <th>Bbs1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>16.92x9.015</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>1</td>
        </tr>
        <tr>
          <th>16.945x11.075</th>
          <td>0</td>
          <td>0</td>
          <td>3</td>
          <td>2</td>
          <td>2</td>
        </tr>
        <tr>
          <th>16.97x10.118</th>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>16.939x12.132</th>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>4</td>
        </tr>
        <tr>
          <th>16.949x13.055</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>3</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    sample_info = pd.read_csv('Analysis/MouseOB/MOB_sample_info.csv', index_col=0)
    counts = counts.loc[sample_info.index]  # Align count matrix with metadata table
    
    sample_info.head(5)




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x</th>
          <th>y</th>
          <th>total_counts</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>16.92x9.015</th>
          <td>16.920</td>
          <td>9.015</td>
          <td>18790</td>
        </tr>
        <tr>
          <th>16.945x11.075</th>
          <td>16.945</td>
          <td>11.075</td>
          <td>36990</td>
        </tr>
        <tr>
          <th>16.97x10.118</th>
          <td>16.970</td>
          <td>10.118</td>
          <td>12471</td>
        </tr>
        <tr>
          <th>16.939x12.132</th>
          <td>16.939</td>
          <td>12.132</td>
          <td>22703</td>
        </tr>
        <tr>
          <th>16.949x13.055</th>
          <td>16.949</td>
          <td>13.055</td>
          <td>18641</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    norm_expr = NaiveDE.stabilize(counts.T).T
    resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T


For the sake of this example, let's just run the test on 1000 random
genes

.. code:: ipython3

    sample_resid_expr = resid_expr.sample(n=1000, axis=1, random_state=24)
    
    X = sample_info[['x', 'y']]
    results = SpatialDE.run(X, sample_resid_expr)


.. parsed-literal::

    INFO:root:Performing DE test
    INFO:root:Pre-calculating USU^T = K's ...
    INFO:root:Done: 0.076s
    INFO:root:Fitting gene models
    INFO:root:Model 1 of 10
    INFO:root:Model 2 of 10                             
    INFO:root:Model 3 of 10                            
    INFO:root:Model 4 of 10                             
    INFO:root:Model 5 of 10                            
    INFO:root:Model 6 of 10                             
    INFO:root:Model 7 of 10                             
    INFO:root:Model 8 of 10                            
    INFO:root:Model 9 of 10                             
    INFO:root:Model 10 of 10                            
                                                        

The result will be a DataFrame with P-values and other relevant values
for each gene.

The most important columns are

-  ``g`` - The name of the gene
-  ``pval`` - The P-value for spatial differential expression
-  ``qval`` - Signifance after correcting for multiple testing
-  ``l`` - A parameter indicating the distance scale a gene changes
   expression over

.. code:: ipython3

    results.head().T




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Gower</th>
          <td>0.999295</td>
          <td>0.999295</td>
          <td>0.999295</td>
          <td>0.999295</td>
          <td>0.999295</td>
        </tr>
        <tr>
          <th>M</th>
          <td>4</td>
          <td>4</td>
          <td>4</td>
          <td>4</td>
          <td>4</td>
        </tr>
        <tr>
          <th>g</th>
          <td>Tinagl1</td>
          <td>Vstm2l</td>
          <td>6330415B21Rik</td>
          <td>Galnt4</td>
          <td>Leng8</td>
        </tr>
        <tr>
          <th>l</th>
          <td>0.402001</td>
          <td>0.402001</td>
          <td>0.402001</td>
          <td>0.402001</td>
          <td>0.402001</td>
        </tr>
        <tr>
          <th>max_delta</th>
          <td>0.00628877</td>
          <td>0.0484324</td>
          <td>0.837928</td>
          <td>0.00806104</td>
          <td>0.975425</td>
        </tr>
        <tr>
          <th>max_ll</th>
          <td>11.5958</td>
          <td>-125.505</td>
          <td>232.757</td>
          <td>91.4048</td>
          <td>-87.1177</td>
        </tr>
        <tr>
          <th>max_mu_hat</th>
          <td>0.025265</td>
          <td>-4.84373</td>
          <td>0.877441</td>
          <td>0.611605</td>
          <td>-1.56887</td>
        </tr>
        <tr>
          <th>max_s2_t_hat</th>
          <td>0.0540122</td>
          <td>19.2345</td>
          <td>0.386566</td>
          <td>0.34356</td>
          <td>1.19909</td>
        </tr>
        <tr>
          <th>model</th>
          <td>SE</td>
          <td>SE</td>
          <td>SE</td>
          <td>SE</td>
          <td>SE</td>
        </tr>
        <tr>
          <th>n</th>
          <td>260</td>
          <td>260</td>
          <td>260</td>
          <td>260</td>
          <td>260</td>
        </tr>
        <tr>
          <th>time</th>
          <td>0.000803947</td>
          <td>0.000534058</td>
          <td>0.000344038</td>
          <td>0.000543118</td>
          <td>0.000329018</td>
        </tr>
        <tr>
          <th>BIC</th>
          <td>-0.948912</td>
          <td>273.252</td>
          <td>-443.272</td>
          <td>-160.567</td>
          <td>196.478</td>
        </tr>
        <tr>
          <th>max_ll_null</th>
          <td>10.697</td>
          <td>-126.502</td>
          <td>232.255</td>
          <td>90.0269</td>
          <td>-87.3105</td>
        </tr>
        <tr>
          <th>LLR</th>
          <td>0.898786</td>
          <td>0.997314</td>
          <td>0.502405</td>
          <td>1.37795</td>
          <td>0.192801</td>
        </tr>
        <tr>
          <th>fraction_spatial_variance</th>
          <td>0.993746</td>
          <td>0.953774</td>
          <td>0.543916</td>
          <td>0.991998</td>
          <td>0.506044</td>
        </tr>
        <tr>
          <th>pval</th>
          <td>0.343107</td>
          <td>0.317961</td>
          <td>0.478445</td>
          <td>0.240451</td>
          <td>0.660595</td>
        </tr>
        <tr>
          <th>qval</th>
          <td>0.977568</td>
          <td>0.977568</td>
          <td>0.977568</td>
          <td>0.977568</td>
          <td>0.977568</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    results.sort_values('qval').head(10)[['g', 'l', 'qval']]




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>g</th>
          <th>l</th>
          <th>qval</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>720</th>
          <td>Cck</td>
          <td>1.135190</td>
          <td>8.802861e-07</td>
        </tr>
        <tr>
          <th>865</th>
          <td>Ptn</td>
          <td>1.907609</td>
          <td>8.162537e-05</td>
        </tr>
        <tr>
          <th>530</th>
          <td>Prokr2</td>
          <td>1.135190</td>
          <td>1.916110e-03</td>
        </tr>
        <tr>
          <th>505</th>
          <td>Nr2f2</td>
          <td>1.135190</td>
          <td>4.790035e-03</td>
        </tr>
        <tr>
          <th>495</th>
          <td>Frzb</td>
          <td>1.135190</td>
          <td>1.317798e-02</td>
        </tr>
        <tr>
          <th>180</th>
          <td>Olfr635</td>
          <td>0.675535</td>
          <td>1.963151e-02</td>
        </tr>
        <tr>
          <th>437</th>
          <td>Map1b</td>
          <td>1.135190</td>
          <td>4.955250e-02</td>
        </tr>
        <tr>
          <th>615</th>
          <td>Agt</td>
          <td>1.135190</td>
          <td>6.470005e-02</td>
        </tr>
        <tr>
          <th>351</th>
          <td>Cpne4</td>
          <td>1.135190</td>
          <td>7.150909e-02</td>
        </tr>
        <tr>
          <th>397</th>
          <td>Sncb</td>
          <td>1.135190</td>
          <td>8.444712e-02</td>
        </tr>
      </tbody>
    </table>
    </div>



We detected a few spatially differentially expressed genes, *Cck* and
*Ptn* for example.

