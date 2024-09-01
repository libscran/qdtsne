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

#ifndef QDTSNE_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

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
 * Useful when the neighbor search is conducted outside of `initialize()`.
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
 * @tparam num_dim_ Number of embedding dimensions.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param[out] Y Pointer to a 2D array with number of rows and columns equal to `num_dim` and `num_points`, respectively.
 * On output, `Y` is filled with random draws from a standard normal distribution. 
 * @param num_points Number of points in the embedding.
 * @param seed Seed for the random number generator.
 */
template<int num_dim_, typename Float_ = double>
void initialize_random(Float_* Y, size_t num_points, int seed = 42) {
    std::mt19937_64 rng(seed);

    size_t total = num_points * num_dim_;
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
 * @tparam num_dim_ Number of embedding dimensions.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param num_points Number of observations.
 * @param seed Seed for the random number generator.
 *
 * @return A vector of length `num_points * num_dim_` containing random draws from a standard normal distribution. 
 */
template<int num_dim_, typename Float_ = double>
std::vector<Float_> initialize_random(size_t num_points, int seed = 42) {
    std::vector<Float_> Y(num_dim_ * num_points);
    initialize_random<num_dim_>(Y.data(), num_points, seed);
    return Y;
}

/**
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function to execute a range of tasks.
 *
 * @param num_workers Number of workers.
 * @param num_tasks Number of tasks.
 * @param run_task_range Function to iterate over a range of tasks within a worker.
 *
 * By default, this is an alias to `subpar::parallelize_range()`.
 * However, if the `QDTSNE_CUSTOM_PARALLEL` function-like macro is defined, it is called instead. 
 * Any user-defined macro should accept the same arguments as `subpar::parallelize_range()`.
 */
template<typename Task_, class Run_>
void parallelize(int num_workers, Task_ num_tasks, Run_ run_task_range) {
#ifndef QDTSNE_CUSTOM_PARALLEL
    // Don't make this nothrow_ = true, there's too many allocations and the
    // derived methods for the nearest neighbors search could do anything...
    subpar::parallelize(num_workers, num_tasks, std::move(run_task_range));
#else
    QDTSNE_CUSTOM_PARALLEL(num_workers, num_tasks, run_task_range);
#endif
}

}

#endif
