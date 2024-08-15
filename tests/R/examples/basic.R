centers <- matrix(rnorm(100, sd = 2), ncol=10)
chosen <- sample(ncol(centers), 50000, replace=TRUE)
z <- matrix(rnorm(length(chosen) * nrow(centers), mean=centers[,chosen,drop=FALSE]), ncol=length(chosen))

library(BiocNeighbors)
res <- findKNN(z, transposed=TRUE, k=90, BNPARAM=AnnoyParam())

init <- matrix(rnorm(length(chosen) * 2), ncol=2)
library(qdtsne)
system.time(ref <- runTsne(res$index, res$distance, init=init))
system.time(maxd <- runTsne(res$index, res$distance, init=init, max.depth=7))
system.time(leaf <- runTsne(res$index, res$distance, init=init, max.depth=7, leaf.approx=TRUE))

par(mfrow=c(1,3))
plot(ref[,1], ref[,2], col=chosen, main="No approximation")
plot(maxd[,1], maxd[,2], col=chosen, main="Maximum depth")
plot(leaf[,1], leaf[,2], col=chosen, main="Leaf approximation")

system.time(ref <- runTsne(res$index, res$distance, init=init, num.threads=4))
system.time(maxd <- runTsne(res$index, res$distance, init=init, max.depth=7, num.threads=4))
system.time(leaf <- runTsne(res$index, res$distance, init=init, max.depth=7, leaf.approx=TRUE, num.threads=4))
