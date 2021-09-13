#ifndef QDTSNE_GAUSSIAN_HPP
#define QDTSNE_GAUSSIAN_HPP

#include "utils.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace qdtsne {

template<typename Index> 
void compute_gaussian_perplexity(NeighborList<Index>& neighbors, double perplexity) {
    constexpr double max_value = std::numeric_limits<double>::max();
    constexpr double tol = 1e-5;

    const size_t N = neighbors.size();
    const size_t K = (N ? neighbors[0].size() : 0);
    const double log_perplexity = std::log(perplexity);

    #pragma omp parallel
    {
        std::vector<double> squared_delta_dist(K);
        std::vector<double> quad_delta_dist(K);
        std::vector<double> output(K);

        #pragma omp for 
        for (size_t n = 0; n < N; ++n){
            double beta = 1.0;
            double min_beta = 0, max_beta = max_value;
            double sum_P = 0;
            auto& current = neighbors[n];

            // We adjust the probabilities by subtracting the first squared
            // distance from everything. This avoids problems with underflow
            // when converting distances to probabilities; it otherwise has no
            // effect on the entropy or even the final probabilities because it
            // just scales all probabilities up/down (and they need to be
            // normalized anyway, so any scaling effect just cancels out).
            const double first = current[0].second * current[0].second;
            for (int m = 1; m < K; ++m) {
                squared_delta_dist[m] = current[m].second * current[m].second - first;
                quad_delta_dist[m] = squared_delta_dist[m] * squared_delta_dist[m];
            }
            output[0] = 1;  

            for (int iter = 0; iter < 200; ++iter) {
                // Apply gaussian kernel. We skip the first value because
                // we effectively normalized it to 1 by subtracting 'first'. 
                for (int m = 1; m < K; ++m) {
                    output[m] = std::exp(-beta * squared_delta_dist[m]); 
                }

                sum_P = std::accumulate(output.begin() + 1, output.end(), 1.0);
                const double prod = std::inner_product(squared_delta_dist.begin() + 1, squared_delta_dist.end(), output.begin() + 1, 0.0);
                const double entropy = beta * (prod / sum_P) + std::log(sum_P);

                const double diff = entropy - log_perplexity;
                if (std::abs(diff) < tol) {
                    break;
                }

                // Attempt a Newton-Raphson search first. 
                bool nr_ok = false;
#ifndef QDTSNE_BETA_BINARY_SEARCH_ONLY
                const double prod2 = std::inner_product(quad_delta_dist.begin() + 1, quad_delta_dist.end(), output.begin() + 1, 0.0);
                const double d1 = - beta / sum_P * (prod2 - prod * prod / sum_P);
                if (d1) {
                    const double alt_beta = beta - (diff / d1); // if it overflows, we should get Inf or -Inf, so the following comparison should be fine.
                    if (alt_beta > min_beta && alt_beta < max_beta) {
                        beta = alt_beta;
                        nr_ok = true;
                    }
                }
#endif

                // Otherwise do a binary search.
                if (!nr_ok) {
                    if (diff > 0) {
                        min_beta = beta;
                        if (max_beta == max_value) {
                            beta *= 2.0;
                        } else {
                            beta = (beta + max_beta) / 2.0;
                        }
                    } else {
                        max_beta = beta;
                        beta = (beta + min_beta) / 2.0;
                    }
                }
            }

            // Row-normalize current row of P.
            auto nIt = current.begin();
            for (auto& o : output) {
                nIt->second = o / sum_P;
                ++nIt;
            }
        }
    }

    return;
}

}

#endif
