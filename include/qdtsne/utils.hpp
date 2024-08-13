#ifndef QDTSNE_UTILS_HPP
#define QDTSNE_UTILS_HPP

/**
 * @file utils.hpp
 *
 * @brief Utilities for running t-SNE.
 */

#include <random>
#include <cmath>
#include <vector>

#include "aarand/aarand.hpp"

namespace qdtsne {

/**
 * A vector of length equal to the number of observations,
 * where each entry contains the indices of and distances to that observation's nearest neighbors.
 *
 * @tparam Index_ Integer type to use for the indices.
 * @tparam Float_ Floating-point type to use for the calculations.
 */
template<typename Index_, typename Float_>
using NeighborList = std::vector<std::vector<std::pair<Index_, Float_> > >;

/**
 * Determines the appropriate number of neighbors, given a perplexity value.
 * Useful when the neighbor search is conducted outside of the `Tsne` class.
 *
 * @param perplexity Perplexity to use in the t-SNE algorithm.
 * @return Number of nearest neighbors to find.
 */
inline int perplexity_to_k(double perplexity) {
    return std::ceil(perplexity * 3);
}

/**
 * Initializes the starting locations of each observation in the embedding.
 * We do so using our own implementation of the Box-Muller transform,
 * to avoid problems with differences in the distribution functions across C++ standard library implementations.
 *
 * @tparam ndim_ Number of embedding dimensions.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param[out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and N`, respectively.
 * On output, `Y` is filled with random draws from a standard normal distribution. 
 * @param N Number of observations.
 * @param seed Seed for the random number generator.
 */
template<int ndim_, typename Float_ = double>
void initialize_random(Float_* Y, size_t N, int seed = 42) {
    std::mt19937_64 rng(seed);

    size_t total = N * ndim_;
    bool odd = total % 2;
    if (odd) {
        --total;
    }

    // Box-Muller gives us two random values at a time.
    for (size_t i = 0; i < total; i += 2) {
        auto paired = aarand::standard_normal<Float_>(rng);
        Y[i] = paired.first;
        Y[i + 1] = paired.second;
    }

    if (odd) {
        // Adding the poor extra for odd total lengths.
        auto paired = aarand::standard_normal<Float_>(rng);
        Y[total] = paired.first;
    }

    return;
}

/**
 * Creates the initial locations of each observation in the embedding. 
 *
 * @tparam ndim_ Number of embedding dimensions.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param N Number of observations.
 * @param seed Seed for the random number generator.
 *
 * @return A vector of length `N * ndim_` containing random draws from a standard normal distribution. 
 */
template<int ndim_, typename Float_ = double>
std::vector<Float_> initialize_random(size_t N, int seed = 42) {
    std::vector<Float_> Y(ndim_ * N);
    initialize_random<ndim_>(Y.data(), N, seed);
    return Y;
}

}

#endif
