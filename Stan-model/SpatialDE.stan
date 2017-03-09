data {
    int<lower=1> N;
    int<lower=1> D;
    vector[D] X[N];
    vector[N] Y;
}
transformed data {
    vector[N] one_vec;
    matrix[N, N] I;

    for (i in 1:N) {
        one_vec[i] = 1.;
    }
    I = diag_matrix(one_vec);
}
parameters {
    real mu;
    real<lower=0> s2_noise;
    real<lower=0> s2_space;
    real<lower=0> lengthscale;
}
model {
    matrix[N, N] K;

    K = cov_exp_quad(X, s2_space, lengthscale);

    Y ~ multi_normal(mu * one_vec, K + s2_noise * I);
}
