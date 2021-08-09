#ifndef QDTSNE_UTILS_HPP
#define QDTSNE_UTILS_HPP

/**
 * @file tsne.hpp
 *
 * @brief Utilities for running t-SNE.
 */

#include <random>

namespace qdtsne {

/**
 * Initializes the starting locations of each observation in the embedding. 
 *
 * @tparam ndim Number of dimensions.
 *
 * @param[out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and N`, respectively.
 * @param N Number of observations.
 * @param seed Seed for the random number generator.
 *
 * @return `Y` is filled with random draws from a standard normal distribution. 
 */
template<int ndim = 2>
void initialize_random(double* Y, size_t N, int seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist(0, 1);
    for (size_t i = 0; i < N * ndim; ++i) {
        Y[i] = dist(rng);
    }
}

/**
 * Creates the initial locations of each observation in the embedding. 
 *
 * @tparam ndim Number of dimensions.
 *
 * @param N Number of observations.
 * @param seed Seed for the random number generator.
 *
 * @return A vector of length `N * ndim` containing random draws from a standard normal distribution. 
 */
template<int ndim = 2>
std::vector<double> initialize_random(size_t N, int seed = 42) {
    std::vector<double> Y(ndim * N);
    initialize_random<ndim>(Y.data(), N, seed);
    return Y;
}

}

#endif
