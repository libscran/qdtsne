#include "Rcpp.h"
#include <algorithm>

#define QDTSNE_BETA_BINARY_SEARCH_ONLY
#define QDTSNE_CUSTOM_NEIGHBORS
#include "qdtsne/tsne.hpp"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_tsne(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances, Rcpp::NumericMatrix init, int iter, int max_depth, int lie_iter, int mom_iter) {
    qdtsne::Tsne runner;
    runner.set_max_iter(iter).set_max_depth(max_depth).set_stop_lying_iter(lie_iter).set_mom_switch_iter(mom_iter);
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
    auto status = runner.run(std::move(neighbors), (double*)output.begin());

    return Rcpp::transpose(output);
}
