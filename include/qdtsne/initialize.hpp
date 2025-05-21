#ifndef QDTSNE_INITIALIZE_HPP
#define QDTSNE_INITIALIZE_HPP

#include "knncolle/knncolle.hpp"

#include <vector>
#include <stdexcept>
#include <cstddef>

#include "Status.hpp"
#include "Options.hpp"
#include "gaussian.hpp"
#include "symmetrize.hpp"

/**
 * @file initialize.hpp
 * @brief Initialize the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @cond
 */
namespace internal {

template<std::size_t num_dim_, typename Index_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(NeighborList<Index_, Float_> nn, Float_ perp, const Options& options) {
    compute_gaussian_perplexity(nn, perp, options.num_threads);
    symmetrize_matrix(nn);
    return Status<num_dim_, Index_, Float_>(std::move(nn), options);
}

}
/**
 * @endcond
 */

/**
 * Initialize the data structures for t-SNE algorithm, given the nearest neighbors of each observation.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the neighbor distances and output embedding.
 *
 * @param neighbors List of indices and distances to nearest neighbors for each observation. 
 * Each observation should have the same number of neighbors, sorted by increasing distance, which should not include itself.
 * @param options Further options.
 * If `Options::infer_perplexity = true`, the perplexity is determined from `neighbors` and the value in `Options::perplexity` is ignored.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<std::size_t num_dim_, typename Index_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(NeighborList<Index_, Float_> neighbors, const Options& options) {
    Float_ perp;
    if (options.infer_perplexity && neighbors.size()) {
        perp = static_cast<Float_>(neighbors.front().size())/3;
    } else {
        perp = options.perplexity;
    }
    return internal::initialize<num_dim_>(std::move(neighbors), perp, options);
}

/**
 * Overload that accepts a neighbor search index and computes the nearest neighbors for each observation,
 * before proceeding with the initialization of the t-SNE algorithm.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Input_ Floating-point type of the input data for the neighbor search.
 * This is not used other than to define the `knncolle::Prebuilt` type.
 * @tparam Float_ Floating-point type of the neighbor distances and output embedding.
 *
 * @param prebuilt A neighbor search index built on the dataset of interest.
 * @param options Further options.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<std::size_t num_dim_, typename Index_, typename Input_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(const knncolle::Prebuilt<Index_, Input_, Float_>& prebuilt, const Options& options) { 
    const int K = perplexity_to_k(options.perplexity); 
    auto neighbors = knncolle::find_nearest_neighbors(prebuilt, K, options.num_threads);
    return internal::initialize<num_dim_>(std::move(neighbors), static_cast<Float_>(options.perplexity), options);
}

/**
 * Overload that accepts a column-major matrix of coordinates and computes the nearest neighbors for each observation,
 * before proceeding with the initialization of the t-SNE algorithm.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the input data, neighbor distances and output embedding.
 * @tparam Matrix_ Class of the input matrix for the neighbor search.
 * This should be a `knncolle::SimpleMatrix` or its base class (i.e., `knncolle::Matrix`).
 *
 * @param data_dim Number of rows of the matrix at `data`, corresponding to the dimensions of the input dataset.
 * @param num_points Number of columns of the matrix at `data`, corresponding to the points of the input dataset.
 * @param[in] data Pointer to an array containing a column-major matrix with `data_dim` rows and `num_points` columns.
 * @param builder A `knncolle::Builder` instance specifying the nearest-neighbor algorithm to use.
 * @param options Further options.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<std::size_t num_dim_, typename Index_, typename Float_, class Matrix_>
Status<num_dim_, Index_, Float_> initialize(
    std::size_t data_dim,
    std::size_t num_points,
    const Float_* data,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    const Options& options) 
{
    auto index = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(data_dim, num_points, data));
    return initialize<num_dim_>(*index, options);
}

}

#endif
