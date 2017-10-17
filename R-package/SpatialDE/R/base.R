# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
#
# Some useful keyboard shortcuts for package authoring:
#
#   Build and Reload Package:  'Cmd + Shift + B'
#   Check Package:             'Cmd + Shift + E'
#   Test Package:              'Cmd + Shift + T'

SE_kernel <- function(X, l) {
  R2 <- as.matrix(dist(X) ** 2)
  K <- exp(-R2 / (2 * l ** 2))
  K
}

factor <- function(K) {
  factors <- eigen(K)
  factors$values[factors$values < 0.] <- 0.
  list('U' = factors$vectors, 'S' = factors$values)
}

get_UT1 <- function(U) {
  colSums(U)
}

get_UTy <- function(U, y) {
  t(y) %*% U
}

mu_hat <- function(delta, UTy, UT1, S) {
  UT1_scaled <- UTy / (S + delta)
  sum1 <- UT1_scaled %*% t(UTy)
  sum2 <- UT1_scaled %*% UT1
  sum1 / sum2
}

LL <- function(delta, UTy, UT1, S, n) {
  mu_h <- mu_hat(delta, UTy, UT1, S)

  sum_1 <- sum((UTy - UT1 * mu_h) / (S + delta))
  sum_2 <- sum(log(S + delta))

  -0.5 * (n * log(2 * pi) + n * log(sum_1 / n) + sum_2 + n)
}

make_objective <- function(UTy, UT1, S, n){
  ll_obj <- function(log_delta){
    -LL(exp(log_delta), UTy, UT1, S, n)
  }
  ll_obj
}
