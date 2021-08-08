# Building the function.
library(Rcpp)
if (!file.exists("qdtsne")) {
    file.symlink("../../include/qdtsne", "qdtsne")
}
sourceCpp("test.cpp")

# Generating some data.
set.seed(10)
mat <- matrix(rnorm(5000), ncol=10)

library(FNN)
res <- FNN::get.knn(mat, k=50)

library(Rtsne)
library(testthat)

# Note that we can't use many iterations here, as 
# divergence happens pretty quickly.
test_that("stats match up", {
    Y <- matrix(rnorm(nrow(mat) * 2), ncol=2)
    ref <- Rtsne_neighbors(res$nn.index, res$nn.dist, Y_init = Y, max_iter = 10, mom_switch_iter=250, stop_lying_iter=250)
    obs <- run_tsne(t(res$nn.index - 1L), t(res$nn.dist), Y, iter = 10, max_depth=100, mom_iter=250, lie_iter=250)
    expect_equal(ref$Y, obs, tol=1e-6)
})

test_that("switch is done correctly", {
    Y <- matrix(rnorm(nrow(mat) * 2), ncol=2)
    ref <- Rtsne_neighbors(res$nn.index, res$nn.dist, Y_init = Y, max_iter = 10, mom_switch_iter=5, stop_lying_iter=5)
    obs <- run_tsne(t(res$nn.index - 1L), t(res$nn.dist), Y, iter = 10, max_depth=100, mom_iter=5, lie_iter=5)
    expect_equal(ref$Y, obs, tol=1e-6)
})
