#ifndef QDTSNE_GAUSSIAN_HPP
#define QDTSNE_GAUSSIAN_HPP

#include "utils.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace qdtsne {

template<typename Index, typename Float = double>
void compute_gaussian_perplexity(NeighborList<Index, Float>& neighbors, Float perplexity, int nthreads) {
    constexpr Float max_value = std::numeric_limits<Float>::max();
    constexpr Float tol = 1e-5;

    const size_t N = neighbors.size();
    const size_t K = (N ? neighbors[0].size() : 0);
    const Float log_perplexity = std::log(perplexity);

#ifndef QDTSNE_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(nthreads)
    {
#else
    QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
#endif
        std::vector<Float> squared_delta_dist(K);
        std::vector<Float> quad_delta_dist(K);
        std::vector<Float> output(K);

#ifndef QDTSNE_CUSTOM_PARALLEL
        #pragma omp for 
        for (size_t n = 0; n < N; ++n){
#else
        for (size_t n = first_; n < last_; ++n) {
#endif

            Float beta = 1.0;
            Float min_beta = 0, max_beta = max_value;
            Float sum_P = 0;
            auto& current = neighbors[n];

            // We adjust the probabilities by subtracting the first squared
            // distance from everything. This avoids problems with underflow
            // when converting distances to probabilities; it otherwise has no
            // effect on the entropy or even the final probabilities because it
            // just scales all probabilities up/down (and they need to be
            // normalized anyway, so any scaling effect just cancels out).
            const Float first2 = current[0].second * current[0].second;
            for (int m = 1; m < K; ++m) {
                // We do an explicit cast here, avoids differences in float vs
                // double intermediate precision that could causes negative
                // squares in the presence of tied distances.
                squared_delta_dist[m] = static_cast<Float>(current[m].second * current[m].second) - first2; 
                quad_delta_dist[m] = squared_delta_dist[m] * squared_delta_dist[m];
            }

            output[0] = 1;  

            for (int iter = 0; iter < 200; ++iter) {
                // Apply gaussian kernel. We skip the first value because
                // we effectively normalized it to 1 by subtracting 'first'. 
                for (int m = 1; m < K; ++m) {
                    output[m] = std::exp(-beta * squared_delta_dist[m]); 
                }

                sum_P = std::accumulate(output.begin() + 1, output.end(), static_cast<Float>(1));
                const Float prod = std::inner_product(squared_delta_dist.begin() + 1, squared_delta_dist.end(), output.begin() + 1, static_cast<Float>(0));
                const Float entropy = beta * (prod / sum_P) + std::log(sum_P);

                const Float diff = entropy - log_perplexity;
                if (std::abs(diff) < tol) {
                    break;
                }

                // Attempt a Newton-Raphson search first. 
                bool nr_ok = false;
#ifndef QDTSNE_BETA_BINARY_SEARCH_ONLY
                const Float prod2 = std::inner_product(quad_delta_dist.begin() + 1, quad_delta_dist.end(), output.begin() + 1, static_cast<Float>(0));
                const Float d1 = - beta / sum_P * (prod2 - prod * prod / sum_P);
                if (d1) {
                    const Float alt_beta = beta - (diff / d1); // if it overflows, we should get Inf or -Inf, so the following comparison should be fine.
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
                            beta *= static_cast<Float>(2);
                        } else {
                            beta = (beta + max_beta) / static_cast<Float>(2);
                        }
                    } else {
                        max_beta = beta;
                        beta = (beta + min_beta) / static_cast<Float>(2);
                    }
                }

                if (std::isinf(beta)) {
                    // Avoid propagation of NaNs via Inf * 0. 
                    for (int m = 1; m < K; ++m) {
                        output[m] = (squared_delta_dist[m] == 0);
                    }
                    break;
                }
            }

            // Row-normalize current row of P.
            auto nIt = current.begin();
            for (auto& o : output) {
                nIt->second = o / sum_P;
                ++nIt;
            }

#ifndef QDTSNE_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    return;
}

}

#endif
