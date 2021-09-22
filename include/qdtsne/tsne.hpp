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
template <int ndim=2>
class Tsne {
public:
    /**
     * @brief Default parameters for t-SNE iterations.
     */
    struct Defaults {
        /**
         * See `set_perplexity()`.
         */
        static constexpr double perplexity = 30;

        /**
         * See `set_infer_perplexity()`.
         */
        static constexpr bool infer_perplexity = true;

        /**
         * See `set_theta()`.
         */
        static constexpr double theta = 0.5;

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
        static constexpr double start_momentum = 0.5;

        /**
         * See `set_final_momentum()`.
         */
        static constexpr double final_momentum = 0.8;

        /**
         * See `set_eta()`.
         */
        static constexpr double eta = 200;

        /**
         * See `set_exaggeration_factor()`.
         */
        static constexpr double exaggeration_factor = 12;

        /**
         * See `set_max_depth()`.
         */
        static constexpr int max_depth = 7;

        /**
         * See `set_interpolation()`.
         */
        static constexpr int interpolation = 0;
    };

private:
    double perplexity = Defaults::perplexity;
    bool infer_perplexity = Defaults::infer_perplexity;
    double theta = Defaults::theta;
    int max_iter = Defaults::max_iter;
    int stop_lying_iter = Defaults::stop_lying_iter;
    int mom_switch_iter = Defaults::mom_switch_iter;
    double start_momentum = Defaults::start_momentum;
    double final_momentum = Defaults::final_momentum;
    double eta = Defaults::eta;
    double exaggeration_factor = Defaults::exaggeration_factor;
    int max_depth = Defaults::max_depth;
    int interpolation = Defaults::interpolation;

public:
    /**
     * @param m Maximum number of iterations to perform.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_max_iter(int m = Defaults::max_iter) {
        max_iter = m;
        return *this;
    }

    /**
     * @param m Maximum depth of the Barnes-Hut tree.
     * Larger values improve the quality of the approximation for the repulsive force calculation, at the cost of computational time.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_max_depth(int m = Defaults::max_depth) {
        max_depth = m;
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
     */
    Tsne& set_mom_switch_iter(int m = Defaults::mom_switch_iter) {
        mom_switch_iter = m;
        return *this;
    }

    /**
     * @param s Starting momentum, to be used in the early iterations before the momentum switch.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_start_momentum(double s = Defaults::start_momentum) {
        start_momentum = s;
        return *this;
    }

    /**
     * @param f Final momentum, to be used in the later iterations after the momentum switch.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_final_momentum(double f = Defaults::final_momentum) {
        final_momentum = f;
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
     */
    Tsne& set_stop_lying_iter(int s = Defaults::stop_lying_iter) {
        stop_lying_iter = s;
        return *this;
    }

    /** 
     * @param e The learning rate, used to scale the updates.
     * Larger values yield larger updates that speed up convergence to a local minima at the cost of stability.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_eta(double e = Defaults::eta) {
        eta = e;
        return *this;
    }

    /** 
     * @param e Factor to scale the probabilities during the early exaggeration phase.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_exaggeration_factor(double e = Defaults::eta) {
        exaggeration_factor = e;
        return *this;
    }

    /**
     * @param p Perplexity, which determines the balance between local and global structure.
     * Higher perplexities will focus on global structure, at the cost of increased runtime and decreased local resolution.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_perplexity(double p = Defaults::perplexity) {
        perplexity = p;
        return *this;
    }

    /**
     * @param i Whether to infer the perplexity in `initialize()` and `run()` methods that accept a `NeighborList` object.
     * In such cases, the value in `set_perplexity()` is ignored.
     * The perplexity is instead defined from the `NeighborList` as the number of nearest neighbors per observation divided by 3.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_infer_perplexity(double i = Defaults::infer_perplexity) {
        infer_perplexity = i;
        return *this;
    }

    /** 
     * @param t Level of the approximation to use in the Barnes-Hut tree calculation of repulsive forces.
     * Lower values increase accuracy at the cost of computational time.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_theta(double t = Defaults::theta) {
        theta = t;
        return *this;
    }

    /**
     * @param i Length of the grid in each dimension when performing interpolation to compute repulsive forces.
     * Larger values improve the resolution of the grid (and the quality of the approximation) at the cost of computational time.
     * A value of 100 is a good compromise for most applications.
     * If set to 0, no interpolation is performed.
     *
     * @return A reference to this `Tsne` object.
     */
    Tsne& set_interpolation(int i = Defaults::interpolation) {
        interpolation = i;
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
        Status(NeighborList<Index> nn, int maxdepth) : 
            neighbors(std::move(nn)),
            N(neighbors.size()),
            dY(N * ndim), 
            uY(N * ndim), 
            gains(N * ndim, 1.0), 
            pos_f(N * ndim), 
            neg_f(N * ndim), 
            tree(N, maxdepth)
#ifdef _OPENMP
            , omp_buffer(N)
#endif
        {
            neighbors.reserve(N);
            return;
        }

        NeighborList<Index> neighbors; 
        const size_t N;
        std::vector<double> dY, uY, gains, pos_f, neg_f;

#ifdef _OPENMP
        std::vector<double> omp_buffer;
#endif

        SPTree<ndim> tree;

        int iter = 0;
        /**
         * @endcond
         */

        /**
         * @return The number of iterations performed on this object so far.
         */
        int iteration() const {
            return iter;
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
    auto initialize(NeighborList<Index> nn) {
        double perp;
        if (infer_perplexity && nn.size()) {
            perp = static_cast<double>(nn.front().size())/3;
        } else {
            perp = perplexity;
        }
        return initialize_internal(std::move(nn), perp);
    }

private:
    template<typename Index = int>
    auto initialize_internal(NeighborList<Index> nn, double perp) {
        Status<typename std::remove_const<Index>::type> status(std::move(nn), max_depth);

#ifdef PROGRESS_PRINTER
        PROGRESS_PRINTER("qdtsne::Tsne::initialize", "Computing neighbor probabilities")
#endif
        compute_gaussian_perplexity(status.neighbors, perp);

#ifdef PROGRESS_PRINTER
        PROGRESS_PRINTER("qdtsne::Tsne::initialize", "Symmetrizing the matrix")
#endif
        symmetrize_matrix(status.neighbors);

        return status;
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
    template<typename Index = int, typename Dist = double>
    auto run(NeighborList<Index> nn, double* Y) {
        auto status = initialize(std::move(nn));
        run(status, Y);
        return status;
    }

    /**
     * @param status The current status of the algorithm, generated either from `initialize()` or from a previous `run()` call.
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn_index.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial locations of each observation; on output, it is updated to the final t-SNE locations.
     *
     * @tparam Index Integer type for the neighbor indices.
     *
     * @return A `Status` object containing the final state of the algorithm after applying iterations.
     */
    template<typename Index = int>
    void run(Status<Index>& status, double* Y) {
        int& iter = status.iter;
        double multiplier = (iter < stop_lying_iter ? exaggeration_factor : 1);
        double momentum = (iter < mom_switch_iter ? start_momentum : final_momentum);

        for(; iter < max_iter; ++iter) {
            // Stop lying about the P-values after a while, and switch momentum
            if (iter == stop_lying_iter) {
                multiplier = 1;
            }
            if (iter == mom_switch_iter) {
                momentum = final_momentum;
            }

            iterate(status, Y, multiplier, momentum);
        }

        return;
    }

private:
    static double sign(double x) { 
        return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
    }

    template<typename Index>
    void iterate(Status<Index>& status,  double* Y, double multiplier, double momentum) {
        compute_gradient(status, Y, multiplier);

        auto& gains = status.gains;
        auto& dY = status.dY;
        auto& uY = status.uY;
        auto& col_P = status.neighbors;

        // Update gains
        for (size_t i = 0; i < gains.size(); ++i) {
            double& g = gains[i];
            g = std::max(0.01, sign(dY[i]) != sign(uY[i]) ? (g + 0.2) : (g * 0.8));
        }

        // Perform gradient update (with momentum and gains)
        for (size_t i = 0; i < gains.size(); ++i) {
            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] += uY[i];
        }

        // Make solution zero-mean
        for (int d = 0; d < ndim; ++d) {
            auto start = Y + d;
            size_t N = col_P.size();

            // Compute means from column-major coordinates.
            double sum = 0;
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
    template<typename Index>
    void compute_gradient(Status<Index>& status, const double* Y, double multiplier) {
        auto& tree = status.tree;
        tree.set(Y);

        compute_edge_forces(status, Y, multiplier);

        size_t N = status.neighbors.size();
        auto& neg_f = status.neg_f;
        std::fill(neg_f.begin(), neg_f.end(), 0);

        double sum_Q = 0;
        if (interpolation) {
            sum_Q = interpolate::compute_non_edge_forces(
                tree, 
                N, 
                Y, 
                theta, 
                neg_f.data(), 
                interpolation
#ifdef _OPENMP
                , status.omp_buffer
#endif
            );

        } else {
#ifdef _OPENMP
            // Don't use reduction methods to ensure that we sum in a consistent order.
            #pragma omp parallel for
            for (size_t n = 0; n < N; ++n) {
                status.omp_buffer[n] = status.tree.compute_non_edge_forces(n, theta, neg_f.data() + n * ndim);
            }
            sum_Q = std::accumulate(status.omp_buffer.begin(), status.omp_buffer.end(), 0.0);
#else
            for (size_t n = 0; n < N; ++n) {
                sum_Q += status.tree.compute_non_edge_forces(n, theta, neg_f.data() + n * ndim);
            }
#endif
        }

        // Compute final t-SNE gradient
        for (size_t i = 0; i < N * ndim; ++i) {
            status.dY[i] = status.pos_f[i] - (neg_f[i] / sum_Q);
        }
    }

    template<typename Index>
    void compute_edge_forces(Status<Index>& status, const double* Y, double multiplier) {
        const auto& neighbors = status.neighbors;
        auto& pos_f = status.pos_f;
        std::fill(pos_f.begin(), pos_f.end(), 0);

        #pragma omp parallel for 
        for (size_t n = 0; n < neighbors.size(); ++n) {
            const auto& current = neighbors[n];
            const double* self = Y + n * ndim;
            double* pos_out = pos_f.data() + n * ndim;

            for (const auto& x : current) {
                double sqdist = 0; 
                const double* neighbor = Y + x.first * ndim;
                for (int d = 0; d < ndim; ++d) {
                    sqdist += (self[d] - neighbor[d]) * (self[d] - neighbor[d]);
                }

                const double mult = multiplier * x.second / (1 + sqdist);
                for (int d = 0; d < ndim; ++d) {
                    pos_out[d] += mult * (self[d] - neighbor[d]);
                }
            }
        }

        return;
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
    template<typename Input = double>
    auto initialize(const Input* input, size_t D, size_t N) { 
#ifdef PROGRESS_PRINTER
        PROGRESS_PRINTER("qdtsne::Tsne::initialize", "Constructing neighbor search indices")
#endif
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
    template<typename Input = double>
    auto run(const Input* input, size_t D, size_t N, double* Y) {
        auto status = initialize(input, D, N);
        run(status, Y);
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
        const int K = std::ceil(perplexity * 3);
        size_t N = searcher->nobs();
        if (K >= N) {
            throw std::runtime_error("number of observations should be greater than 3 * perplexity");
        }

#ifdef PROGRESS_PRINTER
        PROGRESS_PRINTER("qdtsne::Tsne::initialize", "Searching for nearest neighbors")
#endif
        NeighborList<decltype(searcher->nobs())> neighbors(N);

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            neighbors[i] = searcher->find_nearest_neighbors(i, K);
        }

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
    auto run(const Algorithm* searcher, double* Y) {
        auto status = initialize(searcher);
        run(status, Y);
        return status;
    }
};

}

#endif
