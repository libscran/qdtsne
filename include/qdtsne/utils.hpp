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
#include <cstddef>
#include <type_traits>

#include "aarand/aarand.hpp"
#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

#ifndef QDTSNE_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

namespace qdtsne {

/**
 * @brief Lists of neighbors for each observation.
 *
 * This is a convenient alias for the `knncolle::NeighborList` class.
 * Each inner vector corresponds to an observation and contains the list of nearest neighbors for that observation, sorted by increasing distance.
 * Neighbors for each observation should be unique, and the list of neighbors for observation `i` should not contain itself.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the neighbor distances.
 */
template<typename Index_, typename Float_>
using NeighborList = knncolle::NeighborList<Index_, Float_>;

/**
 * Determines the appropriate number of neighbors, given a perplexity value.
 * Useful when the neighbor search is conducted outside of `initialize()`.
 *
 * @tparam Index_ Integer type of the number of neighbors.
 * @param perplexity Perplexity value, see `Options::perplexity`.
 * @return Number of nearest neighbors to find.
 */
template<typename Index_ = int>
Index_ perplexity_to_k(const double perplexity) {
    return sanisizer::from_float<Index_>(std::ceil(perplexity * 3));
}

/**
 * Initializes the starting locations of each observation in the embedding.
 * We do so using our own implementation of the Box-Muller transform,
 * to avoid problems with differences in the distribution functions across C++ standard library implementations.
 *
 * @tparam num_dim_ Number of embedding dimensions.
 * @tparam Float_ Floating-point type of the embedding.
 *
 * @param[out] Y Pointer to a 2D array with number of rows and columns equal to `num_dim` and `num_points`, respectively.
 * On output, `Y` is filled with random draws from a standard normal distribution. 
 * @param num_points Number of points in the embedding.
 * @param seed Seed for the random number generator.
 */
template<std::size_t num_dim_, typename Float_ = double>
void initialize_random(Float_* const Y, const std::size_t num_points, const unsigned long long seed = 42) {
    // The constructor accepts an unsigned type, so any overflow should just wrap around harmlessly. 
    std::mt19937_64 rng(seed);

    // Presumably a size_t can store the product in order to allocate Y in the first place.
    std::size_t num_total = sanisizer::product_unsafe<std::size_t>(num_points, num_dim_);
    const bool odd = num_total % 2;
    if (odd) {
        --num_total;
    }

    // Box-Muller gives us two random values at a time.
    for (std::size_t i = 0; i < num_total; i += 2) {
        auto paired = aarand::standard_normal<Float_>(rng);
        Y[i] = paired.first;
        Y[i + 1] = paired.second;
    }

    if (odd) {
        // Adding the poor extra for odd total lengths.
        auto paired = aarand::standard_normal<Float_>(rng);
        Y[num_total] = paired.first;
    }

    return;
}

/**
 * Creates the initial locations of each observation in the embedding. 
 *
 * @tparam num_dim_ Number of embedding dimensions.
 * @tparam Float_ Floating-point type of the embedding.
 *
 * @param num_points Number of observations.
 * @param seed Seed for the random number generator.
 *
 * @return A vector of length `num_points * num_dim_` containing random draws from a standard normal distribution. 
 */
template<std::size_t num_dim_, typename Float_ = double>
std::vector<Float_> initialize_random(const std::size_t num_points, const unsigned long long seed = 42) {
    std::vector<Float_> Y(sanisizer::product<typename std::vector<Float_>::size_type>(num_points, num_dim_));
    initialize_random<num_dim_>(Y.data(), num_points, seed);
    return Y;
}

/**
 * @tparam Task_ Integer type of the number of tasks.
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
void parallelize(const int num_workers, const Task_ num_tasks, Run_ run_task_range) {
#ifndef QDTSNE_CUSTOM_PARALLEL
    // Don't make this nothrow_ = true, there's too many allocations and the
    // derived methods for the nearest neighbors search could do anything...
    subpar::parallelize(num_workers, num_tasks, std::move(run_task_range));
#else
    QDTSNE_CUSTOM_PARALLEL(num_workers, num_tasks, run_task_range);
#endif
}

/**
 * @cond
 */
template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(const Input_ x) {
    return x;
}
/**
 * @endcond
 */

}

#endif
