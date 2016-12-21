
data {
  int<lower=1> N;
  real x[N];
  vector[N] y;
  vector[N] b;
}
transformed data {
  vector[N] mu;
  for (i in 1:N)
    mu[i] = 0;
}
parameters {
  real<lower=0> s2_se_batch;
  real<lower=0> l_batch;

  real<lower=0> s2_se_global;
  real<lower=0> l_global;

  real<lower=0> s2_model;
}
model {
  matrix[N,N] Sigma;

  // off-diagonal elements
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      Sigma[i,j] = s2_se_global * exp(-pow(x[i] - x[j],2) / (2 * pow(l_global, 2)));
      if (b[i] == b[j]) {
        Sigma[i,j] = Sigma[i,j] + s2_se_batch * exp(-pow(x[i] - x[j],2) / (2 * pow(l_batch, 2)));
      }
      Sigma[j,i] = Sigma[i,j];
    }
  }

  // diagonal elements
  for (k in 1:N)
    Sigma[k,k] = s2_se_batch + s2_se_global + s2_model + 1e-4;

  # Hyper parameter priors
  s2_se_batch ~ cauchy(0, 5);
  l_batch ~ cauchy(0, 5);
  s2_se_global ~ cauchy(0, 5);
  l_global ~ cauchy(0, 5);
  s2_model ~ cauchy(0, 5);

  y ~ multi_normal(mu, Sigma);
}
