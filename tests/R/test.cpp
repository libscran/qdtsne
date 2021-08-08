#include "Rcpp.h"
#include <algorithm>

#define QDTSNE_CUSTOM_NEIGHBORS
#include "qdtsne/tsne.hpp"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_tsne(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances, Rcpp::NumericMatrix init, int iter, int max_depth, int lie_iter, int mom_iter) {
    qdtsne::Tsne runner;
    runner.set_max_iter(iter).set_max_depth(max_depth).set_stop_lying_iter(lie_iter).set_mom_switch_iter(mom_iter);
    int nr = indices.nrow(), nc = indices.ncol();

    std::vector<const int*> ind_ptrs;
    std::vector<const double*> dist_ptrs;
    const int* iptr = indices.begin();
    const double* dptr = distances.begin();
    for (int i = 0; i < nc; ++i, iptr += nr, dptr += nr) {
        ind_ptrs.push_back(iptr);                
        dist_ptrs.push_back(dptr);
    }

    Rcpp::NumericMatrix output(Rcpp::transpose(init));
    auto status = runner.run(ind_ptrs, dist_ptrs, nr, (double*)output.begin());

    return Rcpp::transpose(output);
}
