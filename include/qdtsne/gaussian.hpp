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


/**
 * The aim of this function is to convert distances into probabilities
 * using a Gaussian kernel. Given the following equations:
 *
 * q_i = exp(-beta * (dist_i)^2)
 * p_i = q_i / sum(q_i)
 * entropy = -sum(p_i * log(p_i))
 *         = sum(beta * (dist_i)^2 * q_i) / sum(q_i) + log(sum(q_i))
 *
 * Where the sum is coputed over all neighbors 'i' for each obesrvatino.
 * Our aim is to find 'beta' such that:
 *
 * entropy == target
 *
 * We using Newton's method with a fallback to a binary search if the former
 * doesn't give sensible steps.
 * 
 * NOTE: the QDTSNE_R_PACKAGE_TESTING macro recapitulates the gaussian kernel
 * of the Rtsne package so that we can get a more precise comparison to a
 * trusted reference implementation. It should not be defined in production.
 */

template<bool use_newton_ = 
#ifndef QDTSNE_R_PACKAGE_TESTING
true
#else
false
#endif
, typename Index_, typename Float_>
void compute_gaussian_perplexity(NeighborList<Index_, Float_>& neighbors, Float_ perplexity, int nthreads) {
    Index_ num_points = neighbors.size();
    const Float_ log_perplexity = std::log(perplexity);

    parallelize(nthreads, num_points, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Float_> squared_delta_dist;
        std::vector<Float_> quad_delta_dist;
        std::vector<Float_> prob_numerator; // i.e., the numerator of the probability.

        for (Index_ n = start, end = start + length; n < end; ++n) {
            auto& current = neighbors[n];
            const int K = current.size();
            if (K == 0) {
                continue;
            }

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

            for (int m = 1; m < K; ++m) {
                Float_ dist = current[m].second;
                Float_ squared_delta_dist_raw = dist * dist - first2; 
                squared_delta_dist[m] = squared_delta_dist_raw;
                quad_delta_dist[m] = squared_delta_dist_raw * squared_delta_dist_raw;
            }

            auto last_squared_delta = squared_delta_dist.back();
            if (last_squared_delta == 0) { // quitting early as entropy doesn't depend on beta.
                for (auto& x : current) {
                    x.second = 1.0 / K;
                }
                continue;
            }

            // Choosing an initial beta that matches the scale of the (squared) distances.
            // The choice of numerator is largely based on trial and error to see what
            // minimizes the number of iterations in some simulated data.
            Float_ beta = 
#ifndef QDTSNE_R_PACKAGE_TESTING
                3.0 / last_squared_delta
#else
                1
#endif
            ;

            constexpr Float_ max_value = std::numeric_limits<Float_>::max();
            Float_ min_beta = 0, max_beta = max_value;
            Float_ sum_P = 0;
            prob_numerator.resize(K);
            prob_numerator[0] = 1;

            constexpr int max_iter = 200;
            for (int iter = 0; iter < max_iter; ++iter) {
                // We skip the first value because we know that squared_delta_dist[0] = 0
                // (as we subtracted 'first') and thus prob_numerator[0] = 1. We repeat this for
                // all iterations from [1, K), e.g., squared_delta_dist, quad_delta_dist.
                for (int m = 1; m < K; ++m) {
                    prob_numerator[m] = std::exp(-beta * squared_delta_dist[m]); 
                }

                sum_P = std::accumulate(prob_numerator.begin() + 1, prob_numerator.end(), static_cast<Float_>(1));
                const Float_ prod = std::inner_product(squared_delta_dist.begin() + 1, squared_delta_dist.end(), prob_numerator.begin() + 1, static_cast<Float_>(0));
                const Float_ entropy = beta * (prod / sum_P) + std::log(sum_P);

                const Float_ diff = entropy - log_perplexity;
                constexpr Float_ tol = 1e-5;
                if (std::abs(diff) < tol) {
                    break;
                }

                // Refining the search interval for a (potential) binary search
                // later. We know that the entropy is monotonic decreasing with
                // increasing beta, so if the difference from the target is
                // positive, the current beta must be on the left of the root,
                // and vice versa if the difference is negative.
                if (diff > 0) {
                    min_beta = beta;
                } else {
                    max_beta = beta;
                }

                bool nr_ok = false;
                if constexpr(use_newton_) {
                    // Attempt a Newton-Raphson search first. Note to self: derivative was a bit
                    // painful but pops out nicely enough, use R's D() to prove it to yourself
                    // in the simple case of K = 2 where d0, d1 are the squared deltas.
                    // > D(expression(b * (d0 * exp(- b * d0) + d1 * exp(- b * d1)) / (exp(-b*d0) + exp(-b*d1)) + log(exp(-b*d0) + exp(-b*d1))), name="b")
                    const Float_ prod2 = std::inner_product(quad_delta_dist.begin() + 1, quad_delta_dist.end(), prob_numerator.begin() + 1, static_cast<Float_>(0));
                    const Float_ d1 = - beta / sum_P * (prod2 - prod * prod / sum_P);

                    if (d1) {
                        const Float_ alt_beta = beta - (diff / d1); // if it overflows, we should get Inf or -Inf, so the following comparison should be fine.
                        if (alt_beta > min_beta && alt_beta < max_beta) {
                            beta = alt_beta;
                            nr_ok = true;
                        }
                    }
                }

                if (!nr_ok) {
                    // Doing the binary search, if Newton's failed or was not requested.
                    if (diff > 0) {
                        if (max_beta == max_value) {
                            beta *= static_cast<Float_>(2);
                        } else {
                            beta += (max_beta - beta) / static_cast<Float_>(2); // i.e., midpoint that avoids problems with potential overflow.
                        }
                    } else {
                        beta += (min_beta - beta) / static_cast<Float_>(2); // i.e., midpoint that avoids problems with potential underflow.
                    }
                }

                if (std::isinf(beta)) {
                    // Avoid propagation of NaNs via Inf * 0. 
                    for (int m = 1; m < K; ++m) {
                        prob_numerator[m] = (squared_delta_dist[m] == 0);
                    }
                    sum_P = std::accumulate(prob_numerator.begin(), prob_numerator.end(), static_cast<Float_>(0));
                    break;
                }
            }

            for (int m = 0; m < K; ++m) {
                current[m].second = prob_numerator[m] / sum_P;
            }
        }
    });

    return;
}

}

}

#endif
