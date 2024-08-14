#' @export
runTsne <- function(index, distance, init=NULL, iter=1000, max.depth=100, mom.iter=250, lie.iter=250, leaf.approx=FALSE, num.threads=1) {
    if (is.null(init)) {
        init <- matrix(rnorm(nrow(index) * 2), ncol=2L)
    }

    run_tsne(
        t(index - 1L),
        t(distance),
        init,
        iter=iter,
        max_depth=max.depth,
        mom_iter=mom.iter,
        lie_iter=lie.iter,
        leaf_approx=leaf.approx,
        num_threads=num.threads
    )
}
