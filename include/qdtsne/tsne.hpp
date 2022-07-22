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

#ifndef QDTSNE_TSNE_HPP
#define QDTSNE_TSNE_HPP

#include "gaussian.hpp"
#include "symmetrize.hpp"
#include "sptree.hpp"
#include "interpolate.hpp"

#ifndef QDTSNE_CUSTOM_NEIGHBORS
#include "knncolle/knncolle.hpp"
#endif

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <limits>

/**
 * @file tsne.hpp
 *
 * @brief Implements the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @brief Implements the t-SNE algorithm.
 *
 * The t-distributed stochastic neighbor embedding (t-SNE) algorithm is a non-linear dimensionality reduction technique for visualizing high-dimensional datasets.
 * It places each observation in a low-dimensional map (usually 2D) in a manner that preserves the identity of its neighbors in the original space, thus preserving the local structure of the dataset.
 * This is achieved by converting the distances between neighbors in high-dimensional space to probabilities via a Gaussian kernel;
 * creating a low-dimensional representation where the distances between neighbors can be converted to similar probabilities (in this case, with a t-distribution);
 * and then iterating such that the Kullback-Leiber divergence between the two probability distributions is minimized.
 * In practice, this involves balancing the attractive forces between neighbors and repulsive forces between all points.
 *
 * @tparam Number of dimensions of the final embedding.
 * Values typically range from 2-3. 
 * @tparam Float Floating-point type to use for the calculations.
 *
 * @see
 * van der Maaten, L.J.P. and Hinton, G.E. (2008). 
 * Visualizing high-dimensional data using t-SNE. 
 * _Journal of Machine Learning Research_, 9, 2579-2605.
 *
 * @see 
 * van der Maaten, L.J.P. (2014). 
 * Accelerating t-SNE using tree-based algorithms. 
 * _Journal of Machine Learning Research_, 15, 3221-3245.
 */
template <int ndim = 2, typename Float = double>
class Tsne {
public:
    /**
     * @brief Default parameters for t-SNE iterations.
     */
    struct Defaults {
        /**
         * See `set_perplexity()`.
         */
        static constexpr Float perplexity = 30;

        /**
         * See `set_infer_perplexity()`.
         */
        static constexpr bool infer_perplexity = true;

        /**
         * See `set_theta()`.
         */
        static constexpr Float theta = 0.5;

        /**
         * See `set_max_iter()`.
         */
        static constexpr int max_iter = 1000;

        /**
         * See `set_stop_lying_iter()`.
         */
        static constexpr int stop_lying_iter = 250;

        /**
         * See `set_mom_switch_iter()`.
         */
        static constexpr int mom_switch_iter = 250;

        /**
         * See `set_start_momentum()`.
         */
        static constexpr Float start_momentum = 0.5;

        /**
         * See `set_final_momentum()`.
         */
        static constexpr Float final_momentum = 0.8;

        /**
         * See `set_eta()`.
         */
        static constexpr Float eta = 200;

        /**
         * See `set_exaggeration_factor()`.
         */
        static constexpr Float exaggeration_factor = 12;

        /**
         * See `set_max_depth()`.
         */
        static constexpr int max_depth = 20;

        /**
         * See `set_interpolation()`.
         */
        static constexpr int interpolation = 0;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

private:
    Float perplexity = Defaults::perplexity;
    bool infer_perplexity = Defaults::infer_perplexity;

    struct IterationParameters {
        Float theta = Defaults::theta;
        int max_iter = Defaults::max_iter;
        int stop_lying_iter = Defaults::stop_lying_iter;
        int mom_switch_iter = Defaults::mom_switch_iter;
        Float start_momentum = Defaults::start_momentum;
        Float final_momentum = Defaults::final_momentum;
        Float eta = Defaults::eta;
        Float exaggeration_factor = Defaults::exaggeration_factor;
        int max_depth = Defaults::max_depth;
        int interpolation = Defaults::interpolation;
        int nthreads = Defaults::num_threads;
    };

    IterationParameters iparams;

public:
    /**
     * @param m Maximum number of iterations to perform.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_max_iter(int m = Defaults::max_iter) {
        iparams.max_iter = m;
        return *this;
    }

    /**
     * @param m Number of iterations to perform before switching from the starting momentum to the final momentum.
     *
     * @return A reference to this `Tsne` object.
     *
     * The idea here is that the update to each point includes a small step in the direction of its previous update, i.e., there is some "momentum" from the previous update.
     * This aims to speed up the optimization and to avoid local minima by effectively smoothing the updates.
     * The starting momentum is usually smaller than the final momentum,
     * to give a chance for the points to improve their organization before encouraging iteration to a specific local minima.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     */
    Tsne& set_mom_switch_iter(int m = Defaults::mom_switch_iter) {
        iparams.mom_switch_iter = m;
        return *this;
    }

    /**
     * @param s Starting momentum, to be used in the early iterations before the momentum switch.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_start_momentum(Float s = Defaults::start_momentum) {
        iparams.start_momentum = s;
        return *this;
    }

    /**
     * @param f Final momentum, to be used in the later iterations after the momentum switch.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_final_momentum(Float f = Defaults::final_momentum) {
        iparams.final_momentum = f;
        return *this;
    }

    /**
     * @param s Number of iterations to perform with exaggerated probabilities, as part of the early exaggeration phase.
     *
     * @return A reference to this `Tsne` object.
     *
     * In the early exaggeration phase, the probabilities are multiplied by `m`.
     * This forces the algorithm to minimize the distances between neighbors, creating an embedding containing tight, well-separated clusters of neighboring cells.
     * Because there is so much empty space, these clusters have an opportunity to move around to find better global positions before the phase ends and they are forced to settle down.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     */
    Tsne& set_stop_lying_iter(int s = Defaults::stop_lying_iter) {
        iparams.stop_lying_iter = s;
        return *this;
    }

    /** 
     * @param e The learning rate, used to scale the updates.
     * Larger values yield larger updates that speed up convergence to a local minima at the cost of stability.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_eta(Float e = Defaults::eta) {
        iparams.eta = e;
        return *this;
    }

    /** 
     * @param e Factor to scale the probabilities during the early exaggeration phase.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_exaggeration_factor(Float e = Defaults::eta) {
        iparams.exaggeration_factor = e;
        return *this;
    }

    /**
     * @param p Perplexity, which determines the balance between local and global structure.
     * Higher perplexities will focus on global structure, at the cost of increased runtime and decreased local resolution.
     *
     * This option affects all `run()` methods.
     * It will also affect `initialize()` methods that do not use precomputed neighbor search results.
     * If `initialize()` is called separately from `run()`, the caller should ensure that the same perplexity value is set in both calls.
     *
     * This option affects all methods except if precomputed neighbor search results are supplied _and_ `set_infer_perplexity()` is `true`.
     * In such cases, the perplexity is inferred from the number of neighbors per observation in the supplied search results.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_perplexity(Float p = Defaults::perplexity) {
        perplexity = p;
        return *this;
    }

    /**
     * @param i Whether to infer the perplexity in `initialize()` and `run()` methods that accept a `NeighborList` object.
     * In such cases, the value in `set_perplexity()` is ignored.
     * The perplexity is instead defined from the `NeighborList` as the number of nearest neighbors per observation divided by 3.
     *
     * This option only affects `run()` and `initialize()` methods that use precomputed neighbor search results.
     * All other methods will not respond to this option.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_infer_perplexity(Float i = Defaults::infer_perplexity) {
        infer_perplexity = i;
        return *this;
    }

    /** 
     * @param t Level of the approximation to use in the Barnes-Hut tree calculation of repulsive forces.
     * Lower values increase accuracy at the cost of computational time.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_theta(Float t = Defaults::theta) {
        iparams.theta = t;
        return *this;
    }

    /**
     * @param m Maximum depth of the Barnes-Hut tree.
     * Larger values improve the quality of the approximation for the repulsive force calculation, at the cost of computational time.
     * A value of 7 is a good compromise for most applications.
     *
     * The default is to use a large value, which means that the tree's depth is unbounded for most practical applications.
     * This aims to be consistent with the original implementation of the BH search,
     * but with some protection against duplicate points that would otherwise result in infinite recursion during tree construction.
     * If users are confident that their data contains no duplicates, they can set the depth to arbitrarily large values.
     *
     * This option affects `initialize()` methods and all `run()` methods that do not accept a `Status` object.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_max_depth(int m = Defaults::max_depth) {
        iparams.max_depth = m;
        return *this;
    }

    /**
     * @param i Length of the grid in each dimension when performing interpolation to compute repulsive forces.
     * Larger values improve the resolution of the grid (and the quality of the approximation) at the cost of computational time.
     * A value of 100 is a good compromise for most applications.
     * If set to 0, no interpolation is performed.
     *
     * This option only affects `run()` methods and is not used during `initialize()`.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_interpolation(int i = Defaults::interpolation) {
        iparams.interpolation = i;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_num_threads(int n = Defaults::num_threads) {
        iparams.nthreads = n;
        return *this;
    }

public:
    /**
     * @brief Current status of the t-SNE iterations.
     *
     * @tparam Index Integer type for the neighbor indices.
     *
     * This class holds the precomputed structures required to perform the t-SNE iterations.
     * Users should refrain from interacting with the internals and should only be passing it to the `Tsne::run()` method.
     */
    template<typename Index> 
    struct Status {
        /**
         * @cond
         */
        Status(NeighborList<Index, Float> nn, IterationParameters p) :
            neighbors(std::move(nn)),
            dY(neighbors.size() * ndim), 
            uY(neighbors.size() * ndim), 
            gains(neighbors.size() * ndim, 1.0), 
            pos_f(neighbors.size() * ndim), 
            neg_f(neighbors.size() * ndim), 
            tree(neighbors.size(), p.max_depth),
            parallel_buffer(p.nthreads > 1 ? neighbors.size() : 0),
            iparams(p)
        {}

        NeighborList<Index, Float> neighbors; 
        std::vector<Float> dY, uY, gains, pos_f, neg_f;

        // Buffer to hold parallel-computed results prior to reduction.
        std::vector<Float> parallel_buffer;

        SPTree<ndim, Float> tree;

        IterationParameters iparams;
        int iter = 0;
        /**
         * @endcond
         */

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
        int max_iter() const {
            return iparams.max_iter;
        }

        /**
         * @return The number of observations in the dataset.
         */
        size_t nobs() const {
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
         * `limit` may be greater than `max_iter()`, to run the algorithm for more iterations than specified during construction of this `Status` object.
         */
        void run(Float* Y, int limit) {
            Float multiplier = (iter < iparams.stop_lying_iter ? iparams.exaggeration_factor : 1);
            Float momentum = (iter < iparams.mom_switch_iter ? iparams.start_momentum : iparams.final_momentum);

            for(; iter < limit; ++iter) {
                // Stop lying about the P-values after a while, and switch momentum
                if (iter == iparams.stop_lying_iter) {
                    multiplier = 1;
                }
                if (iter == iparams.mom_switch_iter) {
                    momentum = iparams.final_momentum;
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
        void run(Float* Y) {
            run(Y, iparams.max_iter);
        }

    private:
        static Float sign(Float x) { 
            constexpr Float zero = 0;
            constexpr Float one = 1;
            return (x == zero ? zero : (x < zero ? -one : one));
        }

        void iterate(Float* Y, Float multiplier, Float momentum) {
            compute_gradient(Y, multiplier);

            // Update gains
            for (size_t i = 0; i < gains.size(); ++i) {
                Float& g = gains[i];
                constexpr Float lower_bound = 0.01;
                constexpr Float to_add = 0.2;
                constexpr Float to_mult = 0.8;
                g = std::max(lower_bound, sign(dY[i]) != sign(uY[i]) ? (g + to_add) : (g * to_mult));
            }

            // Perform gradient update (with momentum and gains)
            for (size_t i = 0; i < gains.size(); ++i) {
                uY[i] = momentum * uY[i] - iparams.eta * gains[i] * dY[i];
                Y[i] += uY[i];
            }

            // Make solution zero-mean
            size_t N = nobs();
            for (int d = 0; d < ndim; ++d) {
                auto start = Y + d;

                // Compute means from column-major coordinates.
                Float sum = 0;
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
        Float compute_non_edge_forces() {
            size_t N = nobs();

#if defined(_OPENMP) || defined(QDTSNE_CUSTOM_PARALLEL)
            if (iparams.nthreads > 1) {
                // Don't use reduction methods, otherwise we get numeric imprecision
                // issues (and stochastic results) based on the order of summation.

#ifndef QDTSNE_CUSTOM_PARALLEL
                #pragma omp parallel for num_threads(iparams.nthreads)
                for (size_t n = 0; n < N; ++n) {
#else
                QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
                for (size_t n = first_; n < last_; ++n) {
#endif                

                    parallel_buffer[n] = tree.compute_non_edge_forces(n, iparams.theta, neg_f.data() + n * ndim);

#ifndef QDTSNE_CUSTOM_PARALLEL
                }
#else
                }
                }, iparams.nthreads);
#endif

                return std::accumulate(parallel_buffer.begin(), parallel_buffer.end(), static_cast<Float>(0));
            }
#endif

            Float sum_Q = 0;
            for (size_t n = 0; n < N; ++n) {
                sum_Q += tree.compute_non_edge_forces(n, iparams.theta, neg_f.data() + n * ndim);
            }
            return sum_Q;
        }

        void compute_gradient(const Float* Y, Float multiplier) {
            tree.set(Y);
            compute_edge_forces(Y, multiplier);

            size_t N = nobs();
            std::fill(neg_f.begin(), neg_f.end(), 0);

            Float sum_Q = 0;
            if (iparams.interpolation) {
                Interpolator<ndim, Float> inter;
                inter.set_num_threads(iparams.nthreads);
                sum_Q = inter.compute_non_edge_forces(
                    tree, 
                    N, 
                    Y, 
                    iparams.theta, 
                    neg_f.data(), 
                    iparams.interpolation,
                    parallel_buffer
                );

            } else {
                sum_Q = compute_non_edge_forces();
            }

            // Compute final t-SNE gradient
            for (size_t i = 0; i < N * ndim; ++i) {
                dY[i] = pos_f[i] - (neg_f[i] / sum_Q);
            }
        }

        void compute_edge_forces(const Float* Y, Float multiplier) {
            std::fill(pos_f.begin(), pos_f.end(), 0);

#ifndef QDTSNE_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(iparams.nthreads)
            for (size_t n = 0; n < neighbors.size(); ++n) {
#else
            QDTSNE_CUSTOM_PARALLEL(neighbors.size(), [&](size_t first_, size_t last_) -> void {
            for (size_t n = first_; n < last_; ++n) {
#endif

                const auto& current = neighbors[n];
                const Float* self = Y + n * ndim;
                Float* pos_out = pos_f.data() + n * ndim;

                for (const auto& x : current) {
                    Float sqdist = 0; 
                    const Float* neighbor = Y + x.first * ndim;
                    for (int d = 0; d < ndim; ++d) {
                        sqdist += (self[d] - neighbor[d]) * (self[d] - neighbor[d]);
                    }

                    const Float mult = multiplier * x.second / (static_cast<Float>(1) + sqdist);
                    for (int d = 0; d < ndim; ++d) {
                        pos_out[d] += mult * (self[d] - neighbor[d]);
                    }
                }

#ifndef QDTSNE_CUSTOM_PARALLEL
            }
#else
            }
            }, iparams.nthreads);
#endif

            return;
        }
    };

public:
    /**
     * @param nn List of indices and distances to nearest neighbors for each observation. 
     * Each observation should have the same number of neighbors, sorted by increasing distance, which should not include itself.
     *
     * @tparam Index Integer type for the neighbor indices.
     *
     * @return A `Status` object containing various precomputed structures required for the iterations in `run()`.
     *
     * If `set_infer_perplexity()` is set to `true`, the perplexity is determined from `nn` and the value in `set_perplexity()` is ignored.
     */
    template<typename Index = int>
    auto initialize(NeighborList<Index, Float> nn) {
        Float perp;
        if (infer_perplexity && nn.size()) {
            perp = static_cast<Float>(nn.front().size())/3;
        } else {
            perp = perplexity;
        }
        return initialize_internal(std::move(nn), perp);
    }

private:
    template<typename Index = int>
    Status<Index> initialize_internal(NeighborList<Index, Float> nn, Float perp) {
        compute_gaussian_perplexity(nn, perp, iparams.nthreads);
        symmetrize_matrix(nn);
        return Status<Index>(std::move(nn), iparams);
    }

public:
    /**
     * @param nn List of indices and distances to nearest neighbors for each observation. 
     * Each observation should have the same number of neighbors, sorted by increasing distance, which should not include itself.
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn_index.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial locations of each observation; on output, it is updated to the final t-SNE locations.
     *
     * @tparam Index Integer type for the neighbor indices.
     * @tparam Dist Floating-point type for the neighbor distances.
     *
     * @return A `Status` object containing the final state of the algorithm after all requested iterations are finished.
     *
     * If `set_infer_perplexity()` is set to `true`, the perplexity is determined from `nn` and the value in `set_perplexity()` is ignored.
     */
    template<typename Index = int, typename Dist = Float>
    auto run(NeighborList<Index, Float> nn, Float* Y) {
        auto status = initialize(std::move(nn));
        status.run(Y);
        return status;
    }

public:
#ifndef QDTSNE_CUSTOM_NEIGHBORS
    /**
     * @tparam Input Floating point type for the input data.
     * 
     * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `D` and `N`, respectively.
     * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
     * @param D Number of dimensions.
     * @param N Number of observations.
     *
     * @return A `Status` object containing various pre-computed structures required for the iterations in `run()`.
     *
     * This differs from the other `run()` methods in that it will internally compute the nearest neighbors for each observation.
     * As with the original t-SNE implementation, it will use vantage point trees for the search.
     * See the other `initialize()` methods to specify a custom search algorithm.
     */
    template<typename Input = Float>
    auto initialize(const Input* input, size_t D, size_t N) { 
        knncolle::VpTreeEuclidean<> searcher(D, N, input); 
        return initialize(&searcher);
    }

    /**
     * @tparam Input Floating point type for the input data.
     * 
     * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `D` and `N`, respectively.
     * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
     * @param D Number of dimensions.
     * @param N Number of observations.
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn_index.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial locations of each observation; on output, it is updated to the final t-SNE locations.
     *
     * @return A `Status` object containing the final state of the algorithm after applying all iterations.
     */
    template<typename Input = Float>
    auto run(const Input* input, size_t D, size_t N, Float* Y) {
        auto status = initialize(input, D, N);
        status.run(Y);
        return status;
    }
#endif

    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     *
     * @return A `Status` object containing various pre-computed structures required for the iterations in `run()`.
     *
     * Compared to other `initialize()` methods, this allows users to construct `searcher` with custom search parameters for finer control over the neighbor search.
     */
    template<class Algorithm>
    auto initialize(const Algorithm* searcher) { 
        const int K = perplexity_to_k(perplexity);
        size_t N = searcher->nobs();
        if (K >= N) {
            throw std::runtime_error("number of observations should be greater than 3 * perplexity");
        }

        NeighborList<decltype(searcher->nobs()), Float> neighbors(N);

#ifndef QDTSNE_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(iparams.nthreads)
        for (size_t i = 0; i < N; ++i) {
#else
        QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
        for (size_t i = first_; i < last_; ++i) {
#endif

            neighbors[i] = searcher->find_nearest_neighbors(i, K);

#ifndef QDTSNE_CUSTOM_PARALLEL
        }
#else
        }
        }, iparams.nthreads);
#endif

        return initialize_internal(std::move(neighbors), perplexity);
    }

    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn_index.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial locations of each observation; on output, it is updated to the final t-SNE locations.
     *
     * @return A `Status` object containing the final state of the algorithm after applying all iterations.
     */
    template<class Algorithm> 
    auto run(const Algorithm* searcher, Float* Y) {
        auto status = initialize(searcher);
        status.run(Y);
        return status;
    }
};

}

#endif
