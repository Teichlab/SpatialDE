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
}
model {
    Y ~ multi_normal(mu * one_vec, s2_noise * I);
}
