#ifndef QDTSNE_UTILS_HPP
#define QDTSNE_UTILS_HPP

/**
 * @file utils.hpp
 *
 * @brief Utilities for running t-SNE.
 */

#include <random>
#include <cmath>
#include "aarand/aarand.hpp"

namespace qdtsne {

/**
 * A vector of length equal to the number of observations,
 * where each entry contains the indices of and distances to that observation's nearest neighbors.
 */
template<typename Index>
using NeighborList = std::vector<std::vector<std::pair<Index, double> > >;

/**
 * Initializes the starting locations of each observation in the embedding.
 * We do so using our own implementation of the Box-Muller transform,
 * to avoid problems with differences in the distribution functions across C++ standard library implementations.
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

    size_t total = N * ndim;
    bool odd = total % 2;
    if (odd) {
        --total;
    }

    // Box-Muller gives us two random values at a time.
    for (size_t i = 0; i < total; i += 2) {
        auto paired = aarand::standard_normal(rng);
        Y[i] = paired.first;
        Y[i + 1] = paired.second;
    }

    if (odd) {
        // Adding the poor extra for odd total lengths.
        auto paired = aarand::standard_normal(rng);
        Y[total] = paired.first;
    }

    return;
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
