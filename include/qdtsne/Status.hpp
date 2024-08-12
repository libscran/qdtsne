/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#ifndef QDTSNE_STATUS_HPP
#define QDTSNE_STATUS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <limits>

#include "SPTree.hpp"
#include "Options.hpp"

/**
 * @file Status.hpp
 * @brief Status of the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @brief Current status of the t-SNE iterations.
 *
 * @tparam ndim_ Number of dimensions in the t-SNE embedding.
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Float_ Floating-point type for the distances.
 *
 * This class holds the precomputed structures required to perform the t-SNE iterations.
 * Instances should not be constructed directly but instead created by `initialize()`.
 */
template<int ndim_, typename Index_, typename Float_> 
class Status {
public:
    /**
     * @cond
     */
    Status(NeighborList<Index_, Float_> neighbors, Options options) :
        my_neighbors(std::move(neighbors)),
        my_dY(my_neighbors.size() * ndim_), 
        my_uY(my_neighbors.size() * ndim_), 
        my_gains(my_neighbors.size() * ndim_, 1.0), 
        my_pos_f(my_neighbors.size() * ndim_), 
        my_neg_f(my_neighbors.size() * ndim_), 
        my_tree(my_neighbors.size(), p.max_depth),
        my_parallel_buffer(options.num_threads > 1 ? my_neighbors.size() : 0),
        my_options(std::move(options))
    {}
    /**
     * @endcond
     */

private:
    NeighborList<Index_, Float_> my_neighbors; 
    std::vector<Float_> my_dY, my_uY, my_gains, my_pos_f, my_neg_f;

    SPTree<ndim, Float_> my_tree;
    std::vector<Float> my_parallel_buffer; // Buffer to hold parallel-computed results prior to reduction.

    Options my_options;
    int iter = 0;

public:
    /**
     * @return The number of iterations performed on this object so far.
     */
    int iteration() const {
        return iter;
    }

    /**
     * @return The maximum number of iterations. 
     * This can be modified to `run()` the algorithm for more iterations.
     */
    int max_iterations() const {
        return options.max_iterations;
    }

    /**
     * @return The number of observations in the dataset.
     */
    size_t num_observations() const {
        return neighbors.size();
    }

public:
    /**
     * Run the algorithm to the specified number of iterations.
     * This can be invoked repeatedly with increasing `limit` to run the algorithm incrementally.
     *
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial location of each observation; on output, it is updated to the t-SNE location at the specified number of iterations.
     * @param limit Number of iterations to run up to.
     * The actual number of iterations performed will be the difference between `limit` and `iteration()`, i.e., `iteration()` will be equal to `limit` on completion.
     * `limit` may be greater than `max_iterations()`, to run the algorithm for more iterations than specified during construction of this `Status` object.
     */
    void run(Float_* Y, int limit) {
        Float_ multiplier = (iter < options.stop_lying_iter ? options.exaggeration_factor : 1);
        Float_ momentum = (iter < options.mom_switch_iter ? options.start_momentum : options.final_momentum);

        for(; iter < limit; ++iter) {
            // Stop lying about the P-values after a while, and switch momentum
            if (iter == options.stop_lying_iter) {
                multiplier = 1;
            }
            if (iter == options.mom_switch_iter) {
                momentum = options.final_momentum;
            }

            iterate(Y, multiplier, momentum);
        }
    }

    /**
     * Run the algorithm to the maximum number of iterations.
     * If `run()` has already been invoked with an iteration limit, this method will only perform the remaining iterations required for `iteration()` to reach `max_iter()`.
     * If `iteration()` is already greater than `max_iter()`, this method is a no-op.
     *
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial location of each observation; on output, it is updated to the t-SNE location at the specified number of iterations.
     */
    void run(Float_* Y) {
        run(Y, options.max_iterations);
    }

private:
    static Float_ sign(Float_ x) { 
        constexpr Float_ zero = 0;
        constexpr Float_ one = 1;
        return (x == zero ? zero : (x < zero ? -one : one));
    }

    void iterate(Float_* Y, Float_ multiplier, Float_ momentum) {
        compute_gradient(Y, multiplier);

        // Update gains
        for (size_t i = 0; i < gains.size(); ++i) {
            Float_& g = gains[i];
            constexpr Float_ lower_bound = 0.01;
            constexpr Float_ to_add = 0.2;
            constexpr Float_ to_mult = 0.8;
            g = std::max(lower_bound, sign(dY[i]) != sign(uY[i]) ? (g + to_add) : (g * to_mult));
        }

        // Perform gradient update (with momentum and gains)
        for (size_t i = 0; i < gains.size(); ++i) {
            uY[i] = momentum * uY[i] - options.eta * gains[i] * dY[i];
            Y[i] += uY[i];
        }

        // Make solution zero-mean
        size_t N = num_observations();
        for (int d = 0; d < ndim; ++d) {
            auto start = Y + d;

            // Compute means from column-major coordinates.
            Float_ sum = 0;
            for (size_t i = 0; i < N; ++i, start += ndim) {
                sum += *start;
            }
            sum /= N;

            start = Y + d;
            for (size_t i = 0; i < N; ++i, start += ndim) {
                *start -= sum;
            }
        }

        return;
    }

private:
    Float_ compute_non_edge_forces() {
        size_t N = num_observations();

#if defined(_OPENMP) || defined(QDTSNE_CUSTOM_PARALLEL)
        if (options.num_threads > 1) {
            // Don't use reduction methods, otherwise we get numeric imprecision
            // issues (and stochastic results) based on the order of summation.

#ifndef QDTSNE_CUSTOM_PARALLEL
#ifdef _OPENMP
            #pragma omp parallel num_threads(options.num_threads)
#endif
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for (size_t n = 0; n < N; ++n) {
#else
            QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
                for (size_t n = first_; n < last_; ++n) {
#endif                

                    parallel_buffer[n] = tree.compute_non_edge_forces(n, options.theta, neg_f.data() + n * ndim);

#ifndef QDTSNE_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, options.num_threads);
#endif

            return std::accumulate(parallel_buffer.begin(), parallel_buffer.end(), static_cast<Float_>(0));
        }
#endif

        Float_ sum_Q = 0;
        for (size_t n = 0; n < N; ++n) {
            sum_Q += tree.compute_non_edge_forces(n, options.theta, neg_f.data() + n * ndim);
        }
        return sum_Q;
    }

    void compute_gradient(const Float_* Y, Float_ multiplier) {
        tree.set(Y);
        compute_edge_forces(Y, multiplier);

        size_t N = num_observations();
        std::fill(neg_f.begin(), neg_f.end(), 0);

        Float_ sum_Q = compute_non_edge_forces();

        // Compute final t-SNE gradient
        for (size_t i = 0; i < N * ndim; ++i) {
            dY[i] = pos_f[i] - (neg_f[i] / sum_Q);
        }
    }

    void compute_edge_forces(const Float_* Y, Float_ multiplier) {
        std::fill(pos_f.begin(), pos_f.end(), 0);
        size_t N = num_observations();

#ifndef QDTSNE_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel num_threads(options.num_threads)
#endif
        {
#ifdef _OPENMP
            #pragma omp for 
#endif
            for (size_t n = 0; n < N; ++n) {
#else
        QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
            for (size_t n = first_; n < last_; ++n) {
#endif

                const auto& current = neighbors[n];
                size_t offset = n * static_cast<size_t>(ndim); // cast to avoid overflow.
                const Float_* self = Y + offset;
                Float_* pos_out = pos_f.data() + offset;

                for (const auto& x : current) {
                    Float_ sqdist = 0; 
                    const Float_* neighbor = Y + static_cast<size_t>(x.first) * ndim_; // cast to avoid overflow.
                    for (int d = 0; d < ndim; ++d) {
                        Float_ delta = self[d] - neighbor[d];
                        sqdist += delta * delta;
                    }

                    const Float_ mult = multiplier * x.second / (static_cast<Float_>(1) + sqdist);
                    for (int d = 0; d < ndim; ++d) {
                        pos_out[d] += mult * (self[d] - neighbor[d]);
                    }
                }

#ifndef QDTSNE_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, options.num_threads);
#endif

        return;
    }
};

}

#endif
