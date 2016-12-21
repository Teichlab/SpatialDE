
data {
  int<lower=1> N;
  vector[2] x[N];
  vector[N] y;
}
transformed data {
  vector[N] mu;
  for (i in 1:N)
    mu[i] = 0;
}
parameters {
  real<lower=0> s2_se;
  real<lower=0> l;
  real<lower=0> s2_bias;

  real<lower=0> s2_model;
}
model {
  matrix[N,N] Sigma;

  Sigma = cov_exp_quad(x, s2_se, l) + s2_bias;
  for (i in 1:N)
    Sigma[i, i] = Sigma[i, i] + s2_model;

  # Hyper parameter priors
  s2_se ~ cauchy(0, 5);
  l ~ cauchy(0, 5);
  s2_bias ~ cauchy(0, 5);
  s2_model ~ cauchy(0, 5);

  y ~ multi_normal(mu, Sigma);
}
