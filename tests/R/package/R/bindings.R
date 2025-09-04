#' @export
runTsne <- function(
    index,
    distance,
    init=NULL,
    theta=1,
    iter=1000,
    mom.iter=250,
    start.mom=0.5,
    final.mom=0.8,
    lie.iter=250,
    exaggeration=12,
    eta=200,
    max.depth=100,
    leaf.approx=FALSE,
    num.threads=1)
{
    if (is.null(init)) {
        init <- matrix(rnorm(nrow(index) * 2), ncol=2L)
    }

    run_tsne(
        t(index - 1L),
        t(distance),
        init,
        theta=theta,
        iter=iter,
        lie_iter=lie.iter,
        exaggeration=exaggeration, 
        mom_iter=mom.iter,
        start_mom=start.mom,
        final_mom=final.mom,
        eta=eta,
        max_depth=max.depth,
        leaf_approx=leaf.approx,
        num_threads=num.threads
    )
}
