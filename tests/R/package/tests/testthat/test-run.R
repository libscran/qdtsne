# library(testthat); library(qdtsne); source("test-run.R")

# Generating some data.
set.seed(10)
mat <- matrix(rnorm(5000), ncol=10)
res <- BiocNeighbors::findKNN(mat, k=90) # this needs to be the default perplexity * 3.

# Note that we can't use many iterations here, as divergence happens pretty quickly
# due to some changes in the order of floating-point additions.
library(Rtsne)

test_that("stats match up", {
    Y <- matrix(rnorm(nrow(mat) * 2), ncol=2)
    ref <- Rtsne_neighbors(res$index, res$distance, Y_init=Y, max_iter=10, mom_switch_iter=250, stop_lying_iter=250)
    obs <- runTsne(res$index, res$distance, init=Y, iter=10, max.depth=100, mom.iter=250, exaggeration.iter=250)
    expect_equal(ref$Y, obs$embedding, tol=1e-6)
    # costs don't match up here as qdtsne doesn't consider the exaggeration factor.
})

test_that("switch is done correctly", {
    Y <- matrix(rnorm(nrow(mat) * 2), ncol=2)
    ref <- Rtsne_neighbors(res$index, res$dist, Y_init=Y, max_iter=10, mom_switch_iter=5, stop_lying_iter=5)
    obs <- runTsne(res$index, res$distance, init=Y, iter=10, max.depth=100, mom.iter=5, exaggeration.iter=5)
    expect_equal(ref$Y, obs$embedding, tol=1e-6)
    expect_equal(tail(ref$itercost, 1), obs$cost, tol=1e-6)
})
