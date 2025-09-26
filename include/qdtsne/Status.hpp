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
#include <array>
#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "SPTree.hpp"
#include "Options.hpp"
#include "utils.hpp"

/**
 * @file Status.hpp
 * @brief Status of the t-SNE iterations.
 */

namespace qdtsne {

/**
 * @brief Status of the t-SNE iterations.
 *
 * @tparam num_dim_ Number of dimensions in the t-SNE embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the neighbor distances and output embedding.
 *
 * This class holds the precomputed structures required to perform the t-SNE iterations.
 * Instances should not be constructed directly but instead created by `initialize()`.
 */
template<std::size_t num_dim_, typename Index_, typename Float_> 
class Status {
public:
    /**
     * @cond
     */
    Status(NeighborList<Index_, Float_> neighbors, Options options) :
        my_neighbors(std::move(neighbors)),
        my_tree(sanisizer::cast<internal::SPTreeIndex>(my_neighbors.size()), options.max_depth),
        my_options(std::move(options))
    {
        const auto nobs = sanisizer::cast<Index_>(my_neighbors.size()); // use Index_ to check safety of cast in num_observations().
        const std::size_t buffer_size = sanisizer::product<std::size_t>(nobs, num_dim_);

        sanisizer::resize(my_dY, buffer_size);
        sanisizer::resize(my_uY, buffer_size);
        sanisizer::resize(my_gains, buffer_size, static_cast<Float_>(1));
        sanisizer::resize(my_pos_f, buffer_size);
        sanisizer::resize(my_neg_f, buffer_size);

        if (options.num_threads > 1) {
            sanisizer::resize(my_parallel_buffer, nobs);
        }
    }
    /**
     * @endcond
     */

private:
    NeighborList<Index_, Float_> my_neighbors; 
    std::vector<Float_> my_dY, my_uY, my_gains, my_pos_f, my_neg_f;
    Float_ my_non_edge_sum = 0;

    internal::SPTree<num_dim_, Float_> my_tree;
    std::vector<Float_> my_parallel_buffer; // Buffer to hold parallel-computed results prior to reduction.

    Options my_options;
    int my_iter = 0;

    typename decltype(I(my_tree))::LeafApproxWorkspace my_leaf_workspace;

public:
    /**
     * @return The number of iterations performed on this object so far.
     */
    int iteration() const {
        return my_iter;
    }

    /**
     * @return The maximum number of iterations, as specified in `Options::max_iterations`.
     */
    int max_iterations() const {
        return my_options.max_iterations;
    }

    /**
     * @return The number of observations in the dataset.
     */
    Index_ num_observations() const {
        return my_neighbors.size(); // safety of the cast to Index_ is already checked in the constructor.
    }

#ifndef NDEBUG
    /**
     * @cond
     */
    const auto& get_neighbors() const {
        return my_neighbors;
    }
    /**
     * @endcond
     */
#endif

public:
    /**
     * Run the algorithm to the specified number of iterations.
     * This can be invoked repeatedly with increasing `limit` to run the algorithm incrementally.
     *
     * @param[in, out] Y Pointer to a array containing a column-major matrix with number of rows and columns equal to `num_dim_` and `num_observations()`, respectively.
     * Each row corresponds to a dimension of the embedding while each column corresponds to an observation.
     * On input, this should contain the initial location of each observation; on output, it is updated to the t-SNE location at the specified number of iterations.
     * @param limit Number of iterations to run up to.
     * The actual number of iterations performed will be the difference between `limit` and `iteration()`, i.e., `iteration()` will be equal to `limit` on completion.
     * `limit` may be greater than `max_iterations()`, to run the algorithm for more iterations than specified during construction of this `Status` object.
     */
    void run(Float_* const Y, const int limit) {
        Float_ multiplier = (my_iter < my_options.early_exaggeration_iterations ? my_options.exaggeration_factor : 1);
        Float_ momentum = (my_iter < my_options.momentum_switch_iterations ? my_options.start_momentum : my_options.final_momentum);

        for(; my_iter < limit; ++my_iter) {
            if (my_iter == my_options.early_exaggeration_iterations) {
                multiplier = 1;
            }
            if (my_iter == my_options.momentum_switch_iterations) {
                momentum = my_options.final_momentum;
            }
            iterate(Y, multiplier, momentum);
        }
    }

    /**
     * Run the algorithm to the maximum number of iterations.
     * If `run()` has already been invoked with an iteration limit, this method will only perform the remaining iterations required for `iteration()` to reach `max_iterations()`.
     * If `iteration()` is already greater than `max_iterations()`, this method is a no-op.
     *
     * @param[in, out] Y Pointer to a array containing a column-major matrix with number of rows and columns equal to `num_dim_` and `num_observations()`, respectively.
     * Each row corresponds to a dimension of the embedding while each column corresponds to an observation.
     * On input, this should contain the initial location of each observation; on output, it is updated to the t-SNE location at the maximum number of iterations.
     */
    void run(Float_* const Y) {
        run(Y, my_options.max_iterations);
    }

private:
    static Float_ sign(const Float_ x) { 
        constexpr Float_ zero = 0;
        constexpr Float_ one = 1;
        return (x == zero ? zero : (x < zero ? -one : one));
    }

    void iterate(Float_* const Y, const Float_ multiplier, const Float_ momentum) {
        compute_gradient(Y, multiplier);

        // Update gains
        const auto buffer_size = my_gains.size(); 
        for (decltype(I(buffer_size)) i = 0; i < buffer_size; ++i) {
            Float_& g = my_gains[i];
            constexpr Float_ lower_bound = 0.01;
            constexpr Float_ to_add = 0.2;
            constexpr Float_ to_mult = 0.8;
            g = std::max(lower_bound, sign(my_dY[i]) != sign(my_uY[i]) ? (g + to_add) : (g * to_mult));
        }

        // Perform gradient update (with momentum and gains)
        for (decltype(I(buffer_size)) i = 0; i < buffer_size; ++i) {
            my_uY[i] = momentum * my_uY[i] - my_options.eta * my_gains[i] * my_dY[i];
            Y[i] += my_uY[i];
        }

        // Make solution zero-mean for each dimension
        std::array<Float_, num_dim_> means{};
        const Index_ num_obs = num_observations();
        for (Index_ i = 0; i < num_obs; ++i) {
            for (std::size_t d = 0; d < num_dim_; ++d) {
                means[d] += Y[sanisizer::nd_offset<std::size_t>(d, num_dim_, i)];
            }
        }
        for (std::size_t d = 0; d < num_dim_; ++d) {
            means[d] /= num_obs;
        }
        for (Index_ i = 0; i < num_obs; ++i) {
            for (std::size_t d = 0; d < num_dim_; ++d) {
                Y[sanisizer::nd_offset<std::size_t>(d, num_dim_, i)] -= means[d];
            }
        }

        return;
    }

private:
    void compute_gradient(const Float_* const Y, const Float_ multiplier) {
        my_tree.set(Y);
        compute_edge_forces(Y, multiplier);

        std::fill(my_neg_f.begin(), my_neg_f.end(), 0);
        compute_non_edge_forces();

        const auto buffer_size = my_dY.size();
        for (decltype(I(buffer_size)) i = 0; i < buffer_size; ++i) {
            my_dY[i] = my_pos_f[i] - (my_neg_f[i] / my_non_edge_sum);
        }
    }

    void compute_edge_forces(const Float_* const Y, Float_ multiplier) {
        std::fill(my_pos_f.begin(), my_pos_f.end(), 0);
        const Index_ num_obs = num_observations();

        parallelize(my_options.num_threads, num_obs, [&](const int, const Index_ start, const Index_ length) -> void {
            for (Index_ i = start, end = start + length; i < end; ++i) {
                const auto& current = my_neighbors[i];
                const auto offset = sanisizer::product_unsafe<std::size_t>(i, num_dim_);
                const auto self = Y + offset;
                const auto pos_out = my_pos_f.data() + offset;

                for (const auto& x : current) {
                    Float_ sqdist = 0; 
                    const auto neighbor = Y + sanisizer::product_unsafe<std::size_t>(x.first, num_dim_);
                    for (std::size_t d = 0; d < num_dim_; ++d) {
                        const Float_ delta = self[d] - neighbor[d];
                        sqdist += delta * delta;
                    }

                    const Float_ mult = multiplier * x.second / (static_cast<Float_>(1) + sqdist);
                    for (std::size_t d = 0; d < num_dim_; ++d) {
                        pos_out[d] += mult * (self[d] - neighbor[d]);
                    }
                }
            }
        });

        return;
    }

    void compute_non_edge_forces() {
        if (my_options.leaf_approximation) {
            my_tree.compute_non_edge_forces_for_leaves(my_options.theta, my_leaf_workspace, my_options.num_threads);
        }

        const Index_ num_obs = num_observations();
        if (my_options.num_threads > 1) {
            // Don't use reduction methods, otherwise we get numeric imprecision
            // issues (and stochastic results) based on the order of summation.
            parallelize(my_options.num_threads, num_obs, [&](const int, const Index_ start, const Index_ length) -> void {
                for (Index_ i = start, end = start + length; i < end; ++i) {
                    const auto neg_ptr = my_neg_f.data() + sanisizer::product_unsafe<std::size_t>(i, num_dim_);
                    if (my_options.leaf_approximation) {
                        my_parallel_buffer[i] = my_tree.compute_non_edge_forces_from_leaves(i, neg_ptr, my_leaf_workspace);
                    } else {
                        my_parallel_buffer[i] = my_tree.compute_non_edge_forces(i, my_options.theta, neg_ptr);
                    }
                }
            });

            my_non_edge_sum = std::accumulate(my_parallel_buffer.begin(), my_parallel_buffer.end(), static_cast<Float_>(0));
            return;
        }

        my_non_edge_sum = 0;
        for (Index_ i = 0; i < num_obs; ++i) {
            const auto neg_ptr = my_neg_f.data() + sanisizer::product_unsafe<std::size_t>(i, num_dim_);
            if (my_options.leaf_approximation) {
                my_non_edge_sum += my_tree.compute_non_edge_forces_from_leaves(i, neg_ptr, my_leaf_workspace);
            } else {
                my_non_edge_sum += my_tree.compute_non_edge_forces(i, my_options.theta, neg_ptr);
            }
        }
    }

public:
    /**
     * @param[in] Y Pointer to a array containing a column-major matrix with number of rows and columns equal to `num_dim_` and `num_observations()`, respectively.
     * This should contain the location of each observation. 
     *
     * @return The Kullback-Leibler divergence for the current embedding.
     *
     * This method is not `const` as it re-uses some of the pre-allocated buffers in the `Status` object for efficiency.
     * Calling `cost()` at any time will not affect the results of subsequent calls to `run()`.
     *
     * This method does not consider any exaggeration of the conditional probabilities for iterations at or before `Options::early_exaggeration_iterations`.
     * That is, all probabilities used here will not be exaggerated, regardless of the iteration.
     */
    Float_ cost(const Float_* const Y) {
        my_tree.set(Y);
        std::fill(my_neg_f.begin(), my_neg_f.end(), 0);
        compute_non_edge_forces();

        const Index_ num_obs = num_observations();
        Float_ total = 0;
        for (Index_ i = 0; i < num_obs; ++i) {
            const auto& cur_neighbors = my_neighbors[i];
            const auto self = Y + sanisizer::product_unsafe<std::size_t>(i, num_dim_);

            for (const auto& x : cur_neighbors) {
                const auto neighbor = Y + sanisizer::product_unsafe<std::size_t>(x.first, num_dim_);
                Float_ sqdist = 0;
                for (std::size_t d = 0; d < num_dim_; ++d) {
                    const Float_ delta = self[d] - neighbor[d];
                    sqdist += delta * delta;
                }

                const Float_ qprob = (static_cast<Float_>(1) / (static_cast<Float_>(1) + sqdist)) / my_non_edge_sum;
                constexpr Float_ lim = std::numeric_limits<Float_>::min();
                total += x.second * std::log(std::max(lim, x.second) / std::max(lim, qprob));
            }
        }

        return total;
    }
};

}

#endif
