#ifndef QDTSNE_UTILS_HPP
#define QDTSNE_UTILS_HPP

/**
 * @file tsne.hpp
 *
 * @brief Utilities for running t-SNE.
 */

#include <random>
#include <cmath>

namespace qdtsne {

template<class Engine>
double uniform01 (Engine& eng) {
    // Stolen from Boost.
    const double factor = 1.0 / static_cast<double>((eng.max)()-(eng.min)());
    double result;
    do {
        result = static_cast<double>(eng() - (eng.min)()) * factor;
    } while (result == 1.0);
    return result;
}

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
    constexpr double pi = 3.14159265358979323846;

    size_t total = N * ndim;
    bool odd = total % 2;
    if (odd) {
        --total;
    }

    // Box-Muller gives us two random values at a time.
    for (size_t i = 0; i < total; i += 2) {
        double constant = std::sqrt(-2 * std::log(uniform01(rng)));
        double angle = 2 * pi * uniform01(rng);
        Y[i] = constant * std::sin(angle);
        Y[i + 1] = constant * std::cos(angle);
    }

    if (odd) {
        // Adding the poor extra for odd total lengths.
        double constant = std::sqrt(-2 * std::log(uniform01(rng)));
        double angle = 2 * pi * uniform01(rng);
        Y[total] = constant * std::sin(angle);
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
