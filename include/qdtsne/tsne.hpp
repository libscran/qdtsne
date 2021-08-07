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

#include "sptree.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace qdtsne {

template <int ndim=2>
class Tsne {
public: 
    struct Defaults {
        static constexpr double perplexity = 30;
        static constexpr double theta = 0.5;
        static constexpr int max_iter = 1000;
        static constexpr int stop_lying_iter = 250;
        static constexpr int mom_switch_iter = 250;
        static constexpr double start_momentum = 0.5;
        static constexpr double final_momentum = 0.8;
        static constexpr double eta = 200;
        static constexpr double exaggeration_factor = 12;
    };

private:
    double perplexity = Defaults::perplexity;
    double theta = Defaults::theta;
    int max_iter = Defaults::max_iter;
    int stop_lying_iter = Defaults::stop_lying_iter;
    int mom_swith_iter = Defaults::mom_switch_iter;
    double start_momentum = Defaults::start_momentum;
    double final_momentum = Defaults::final_momentum;
    double eta = Defaults::eta;
    double exaggeration_factor = Defaults::exaggeration_factor;

public:
    template<int ndim, typename Index, class Tree>
    struct Status {
        Status(size_t N) : dY(N * ndim), uY(N * ndim), gains(N * ndim, 1.0), pos_f(N * ndim), neg_f(N * ndim), tree(N) {
            neighbors.reserve(N);
            probabilities.reserve(N);
            return;
        }

    private:
        std::vector<std::vector<Index> > neighbors;
        std::vector<std::vector<double> > probabilities;
        std::vector<double> dY, uY, gains, pos_f, neg_f;

#ifdef _OPENMP
        std::vector<double> omp_buffer;
#endif

        Tree tree;
        int iteration = 0;
    };

public:
    template<class Tree = SPTree<ndim, 7>, typename Index = int, typename Dist = double>
    Status<ndim, Index, Tree> run(const std::vector<Index*>& nn_index, const std::vector<Dist*>& nn_dist, int K, double* Y) {
        if (nn_index.size() != nn_dist.size()) {
            throw std::runtime_error("indices and distances should be of the same length");
        }

        Status<ndim, Index, Tree> status(nn_index.size());
        compute_gaussian_perplexity(nn_dist, K, status.probabilities);
        symmetrize_matrix(nn_index, K, status.neighbors, status.probabilities);
        train_iterations(status.neighbors, status.probabilities, 
                Y,
                status.dY, status.dU, status.gains, status.iteration 
#ifdef _OPENMP
                , status.omp_buffer
#endif
                );
        return status;
    }

private:
    template<int ndims, typename Index, typename Dist>
    void compute_gaussian_perplexity(const std::vector<Dist*>& nn_dist, int K, std::vector<std::vector<double> >& val_P) {
        if (perplexity > K) {
            throw std::runtime_error("Perplexity should be lower than K!\n");
        }

        const size_t N = nn_dist.size();
        constexpr double min_value = std::numeric_limits<double>::lowest(); // not min()! surprise!
        constexpr double max_value = std::numeric_limits<double>::max();
        constexpr double tol = 1e-5;
        const double log_perplexity = std::log(perplexity);

        #pragma omp parallel for 
        for (size_t n = 0; n < N; ++n){
            bool found = false;
            double beta = 1.0;
            double min_beta = min_value, max_beta = max_value;
            double sum_P = 0;
            const double* distances = nn_dist + n * K;
            std::vector<double> output(K);

            // Iterate until we found a good perplexity
            for (int iter = 0; iter < 200 && !found; ++iter) {
                for (int m = 0; m < K; ++m) {
                    output[m] = std::exp(-beta * distances[m] * distances[m]); // apply gaussian kernel
                }

                // Compute entropy of current row
                sum_P = std::accumulate(output.begin(), output.end(), min_value);
                double H = .0;
                for(int m = 0; m < K; ++m) {
                    H += beta * distances[m] * distances[m] * output[m];
                }
                H = (H / sum_P) + std::log(sum_P);

                // Evaluate whether the entropy is within the tolerance level
                double Hdiff = H - log_perplexity;
                if (Hdiff < tol && -Hdiff < tol) {
                    found = true;
                } else {
                    if (Hdiff > 0) {
                        min_beta = beta;
                        if (max_beta == max_value || max_beta == min_value) {
                            beta *= 2.0;
                        } else {
                            beta = (beta + max_beta) / 2.0;
                        }
                    } else {
                        max_beta = beta;
                        if (min_beta == max_value || min_beta == min_value) {
                            beta /= 2.0;
                        } else {
                            beta = (beta + min_beta) / 2.0;
                        }
                    }
                }
            }

            // Row-normalize current row of P.
            for (auto& o : output) {
                o /= sum_P;
            }

            val_P.push_back(std::move(output));
        }
        return val_P; 
    }

private:
    template <typename Index>
    void symmetrize_matrix(const std::vector<Index*>& nn_index, int K, std::vector<std::vector<Index> >& col_P, std::vector<double>& probabilities) {
        const size_t N = nn_index.size();

        // Initializing the output neighbor list.
        for (size_t n = 0; n < N; ++n) {
            col_P.emplace_back(nn_index[n], nn_index[n] + K);
        }

        for (size_t n = 0; n < N; ++n) {
            auto my_neighbors = nn_index[n];

            for (int k1 = 0; k1 < K; ++k1) {
                auto curneighbor = my_neighbors[k1];
                auto neighbors_neighbors = nn_index[curneighbor];

                // Check whether the current point is present in its neighbor's set.
                bool present = false;
                int k2 = 0;
                for (; k2 < K; ++k) {
                    if (neighbors_neighbors[k2] == n) {
                        present = true;
                        break;
                    }
                }

                if (!present) {
                    // If not present, no addition of probabilities is involved.
                    col_P[curneighbor].push_back(n);
                    probabilities[curneighbor].push_back(probabilities[n][k1]);
                } else if (n < curneighbor) {
                    // Adding the probabilities - but if n >= curneighbor, then this would have
                    // already been done at n = curneighbor, so we skip this to avoid adding it twice.
                    double sum = probabilities[n][k1] + probabilities[curneighbor][k2];
                    probabilities[n][k1] = sum;
                    probabilities[curneighbor][k2] = sum;
                }
            }
        }

        // Divide the result by two
        double total = 0;
        for (auto& x : probabilities) {
            for (auto& y : x) {
                y /= 2;
                total += y;
            }
        }

        // Probabilities across the entire matrix sum to unity.
        for (auto& x : probabilities) {
            for (auto& y : x) {
                y /= total;
            }
        }

        return col_P;
    }

private:
    static double sign(double x) { 
        return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
    }

    template<int ndim, typename Index>
    void train_iterations(const std::vector<std::vector<Index> >& col_P, const std::vector<std::vector<double> >& val_P, 
        double* Y, int& iter,
        std::vector<double>& dY, std::vector<double>& uY, std::vector<double>& gains, 
        std::vector<double>& pos_f, std::vector<double>& neg_f
#ifdef _OPENMP
        , std::vector<double>& omp_buffer        
#endif        
        )
    {
        const size_t N = col_P.size();
        double multiplier = exaggeration_factor; // Lie about the P-values
        double momentum = start_momentum;
        Tree tree(ndim, N);

        for(; iter < max_iter; ++iter) {
            // Stop lying about the P-values after a while, and switch momentum
            if (iter == stop_lying_iter) {
                multiplier = 1;
            }
            if (iter == mom_switch_iter) {
                momentum = final_momentum;
            }

            iterate(col_P, val_P, Y, dY, uY, gains, pos_f, neg_f, multiplier, momentum
#ifdef _OPENMP
                , omp_buffer        
#endif        
            );
        }
    }

    template<int ndim, typename Index>
    void iterate(const std::vector<std::vector<Index> >& col_P, const std::vector<std::vector<double> >& val_P, 
        double* Y,
        std::vector<double>& dY, std::vector<double>& uY, std::vector<double>& gains, 
        std::vector<double>& pos_f, std::vector<double>& neg_f, 
        double multiplier, double momentum
#ifdef _OPENMP
        , std::vector<double>& omp_buffer
#endif        
        )
    {
        compute_gradient<ndim>(col_P, val_P, Y, dY, uY, gains, pos_f, neg_f, multiplier
#ifdef _OPENMP
            , omp_buffer
#endif
        );

        // Update gains
        for (size_t i = 0; i < gains.size(); i++) {
            double& g = gains[i];
            g = std::max(0.01, (sign(dY[i]) != sign(uY[i])) ? (g + 0.2) : (g * 0.8));
        }

        // Perform gradient update (with momentum and gains)
        for (size_t i = 0; i < gains.size(); ++i) {
            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] += uY[i];
        }

        // Make solution zero-mean
        for (int d = 0; d < ndim; ++d) {
            auto start = Y + d * N;

            // Compute means from column-major coordinates.
            double sum = 0;
            for (size_t i = 0; i < N; ++i, start += ndim) {
                sum += *start;
            }
            sum /= N;

            start = Y + d * N;
            for (size_t i = 0; i < N; ++i, start += ndim) {
                *start /= sum;
            }
        }

        return;
    }

private:
    template<typename Index, class Tree>
    static void compute_gradient(const std::vector<std::vector<Index> >& col_P, const std::vector<std::vector<double> >& val_P, 
        double* Y, std::vector<double>& dC, Tree& tree, std::vector<double>& pos_f, std::vector<double>& neg_f, 
        double multiplier
#ifdef _OPENMP
        , std::vector<double>& omp_buffer
#endif        
    {
        tree.set(Y);

        compute_edge_forces(col_P, val_P, Y, pos_f, multiplier);

        size_t N = col_P.size();
#ifdef _OPENMP
        std::vector<double> output(N);
        #pragma omp parallel for
        for (size_t n = 0; n < N; ++n) {
            output[n] = tree->compute_non_edge_forces(n, theta, neg_f + n * ndim);
        }
        double sum_Q = std::accumulate(output.begin(), output.end(), 0.0);
#else
        double sum_Q = 0;
        for (size_t n = 0; n < N; ++n) {
            sum_Q += tree->compute_non_edge_forces(n, theta, neg_f + n * ndim);
        }
#endif

        // Compute final t-SNE gradient
        for (size_t n = 0; n < N; ++n) {
            dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
        }
    }

    template<typename Index>
    static void compute_edge_forces(const std::vector<std::vector<Index> >& col_P, const std::vector<std::vector<double> >& val_P, 
        const double* Y, std::vector<double>& pos_f, double multiplier) const 
    {
        std::fill(pos_f.begin(), pos_f.end(), 0);                

        #pragma omp parallel for 
        for (size_t n = 0; n < col_P.size(); ++n) {
            const auto& cur_prob = val_P[n];
            const auto& cur_col = col_P[n];
            const double* self = point + n * ndim;
            double* pos_out = pos_f.data() + n * ndim;

            for (size_t i = 0; i < cur_col.size(); ++i) {
                double sqdist = 0; 
                const double* neighbor = point + cur_col[i] * ndim;
                for (int d = 0; d < ndim; ++d) {
                    sqdist += (self[d] - neighbor[d]) * (self[d] - neighbor[d]):
                }

                const double mult = multiplier * cur_prob[i] / (1 + sqdist);
                for (int d = 0; d < ndim; ++d) {
                    pos_out[d] += mult * (self[d] - neighbor[d]);
                }
            }
        }

        return;
    }
};


}

#endif
