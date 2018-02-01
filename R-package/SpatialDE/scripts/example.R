N <- 50

X <- runif(N)

l <- 0.2

K <- SE_kernel(X, l)

fac <- factor(K)

UT1 <- get_UT1(fac$U)

y <- rnorm(N)

UTy <- get_UTy(fac$U, y)

result <- lengthscale_fit(UTy, UT1, fac$S, N)


