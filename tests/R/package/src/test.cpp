#include "Rcpp.h"

#define QDTSNE_R_PACKAGE_TESTING
#include "qdtsne/qdtsne.hpp"

// [[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_tsne(
    Rcpp::IntegerMatrix indices,
    Rcpp::NumericMatrix distances,
    Rcpp::NumericMatrix init,
    double theta,
    int iter,
    int lie_iter,
    double exaggeration,
    int mom_iter,
    double eta,
    double start_mom,
    double final_mom,
    int max_depth,
    bool leaf_approx,
    int num_threads) 
{
    qdtsne::Options opt;
    opt.theta = theta;
    opt.max_iterations = iter;
    opt.stop_lying_iter = lie_iter;
    opt.exaggeration_factor = exaggeration;
    opt.mom_switch_iter = mom_iter;
    opt.start_momentum = start_mom;
    opt.final_momentum = final_mom;
    opt.eta = eta;
    opt.max_depth = max_depth;
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
