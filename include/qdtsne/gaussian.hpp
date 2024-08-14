#ifndef QDTSNE_GAUSSIAN_HPP
#define QDTSNE_GAUSSIAN_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>

#include "utils.hpp"

namespace qdtsne {

namespace internal {

template<typename Index_, typename Float_>
void compute_gaussian_perplexity(NeighborList<Index_, Float_>& neighbors, Float_ perplexity, [[maybe_unused]] int nthreads) {
    constexpr Float_ max_value = std::numeric_limits<Float_>::max();
    constexpr Float_ tol = 1e-5;

    const size_t num_points  = neighbors.size();
    const Float_ log_perplexity = std::log(perplexity);

#ifndef QDTSNE_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#else
    QDTSNE_CUSTOM_PARALLEL(num_points, [&](size_t first_, size_t last_) -> void {
#endif

        std::vector<Float_> squared_delta_dist;
        std::vector<Float_> quad_delta_dist;
        std::vector<Float_> output;

#ifndef QDTSNE_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp for 
#endif
        for (size_t n = 0; n < num_points; ++n){
#else
        for (size_t n = first_; n < last_; ++n) {
#endif

            auto& current = neighbors[n];
            const int K = current.size();
            if (K) {
                squared_delta_dist.resize(K);
                quad_delta_dist.resize(K);

                // We adjust the probabilities by subtracting the first squared
                // distance from everything. This avoids problems with underflow
                // when converting distances to probabilities; it otherwise has no
                // effect on the entropy or even the final probabilities because it
                // just scales all probabilities up/down (and they need to be
                // normalized anyway, so any scaling effect just cancels out).
                const Float_ first = current[0].second;
                const Float_ first2 = first * first;

#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int m = 1; m < K; ++m) {
                    Float_ dist = current[m].second;
                    Float_ squared_delta_dist_raw = dist * dist - first2; 
                    squared_delta_dist[m] = squared_delta_dist_raw;
                    quad_delta_dist[m] = squared_delta_dist_raw * squared_delta_dist_raw;
                }

                Float_ beta = 1.0;
                Float_ min_beta = 0, max_beta = max_value;
                Float_ sum_P = 0;
                output.resize(K);
                output[0] = 1;  

                for (int iter = 0; iter < 200; ++iter) {
                    // We skip the first value because we know that squared_delta_dist[0] = 0
                    // (as we subtracted 'first') and thus output[0] = 1.
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (int m = 1; m < K; ++m) {
                        output[m] = std::exp(-beta * squared_delta_dist[m]); 
                    }

                    sum_P = std::accumulate(output.begin() + 1, output.end(), static_cast<Float_>(1));
                    const Float_ prod = std::inner_product(squared_delta_dist.begin() + 1, squared_delta_dist.end(), output.begin() + 1, static_cast<Float_>(0));
                    const Float_ entropy = beta * (prod / sum_P) + std::log(sum_P);

                    const Float_ diff = entropy - log_perplexity;
                    if (std::abs(diff) < tol) {
                        break;
                    }

                    // Attempt a Newton-Raphson search first. Note to self: derivative was a bit
                    // painful but pops out nicely enough, use R's D() to prove it to yourself
                    // in the simple case of K = 2:
                    // > D(expression((b*c*exp(-b*c) + b*d*exp(-b*d)) / (exp(-b*c) + exp(-b*d)) + log(exp(-b*c) + exp(-b*d))), name="b")
                    bool nr_ok = false;
#ifndef QDTSNE_BETA_BINARY_SEARCH_ONLY
                    const Float_ prod2 = std::inner_product(quad_delta_dist.begin() + 1, quad_delta_dist.end(), output.begin() + 1, static_cast<Float_>(0));
                    const Float_ d1 = - beta / sum_P * (prod2 - prod * prod / sum_P);
                    if (d1) {
                        const Float_ alt_beta = beta - (diff / d1); // if it overflows, we should get Inf or -Inf, so the following comparison should be fine.
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
                                beta *= static_cast<Float_>(2);
                            } else {
                                beta = (beta + max_beta) / static_cast<Float_>(2);
                            }
                        } else {
                            max_beta = beta;
                            beta = (beta + min_beta) / static_cast<Float_>(2);
                        }
                    }

                    if (std::isinf(beta)) {
                        // Avoid propagation of NaNs via Inf * 0. 
#ifdef _OPENMP
                        #pragma omp simd
#endif
                        for (int m = 1; m < K; ++m) {
                            output[m] = (squared_delta_dist[m] == 0);
                        }
                        break;
                    }
                }

                // Row-normalize current row of P.
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int m = 0; m < K; ++m) {
                    current[m].second = output[m] / sum_P;
                }
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

}

#endif
