#include "Rcpp.h"

#define QDTSNE_R_PACKAGE_TESTING
#include "qdtsne/qdtsne.hpp"

// [[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_tsne(
    Rcpp::IntegerMatrix indices,
    Rcpp::NumericMatrix distances,
    Rcpp::NumericMatrix init,
    int iter,
    int max_depth,
    int lie_iter,
    int mom_iter,
    bool leaf_approx,
    int num_threads) 
{
    qdtsne::Options opt;
    opt.max_iterations = iter;
    opt.max_depth = max_depth;
    opt.stop_lying_iter = lie_iter;
    opt.mom_switch_iter = mom_iter;
    opt.leaf_approximation = leaf_approx;
    opt.num_threads = num_threads;

    int nr = indices.nrow(), nc = indices.ncol();
    qdtsne::NeighborList<int, double> neighbors(nc);
    for (int i = 0; i < nc; ++i) {
        auto icol = indices.column(i);
        auto dcol = distances.column(i);
        for (int j = 0; j < nr; ++j) {
            neighbors[i].emplace_back(icol[j], dcol[j]);
        }
    }

    Rcpp::NumericMatrix output(Rcpp::transpose(init));

    auto status = qdtsne::initialize<2>(std::move(neighbors), opt);
    status.run(static_cast<double*>(output.begin()));

    return Rcpp::transpose(output);
}
