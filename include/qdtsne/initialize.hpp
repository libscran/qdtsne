#ifndef QDTSNE_INITIALIZE_HPP
#define QDTSNE_INITIALIZE_HPP

#include <cstddef>

#include "knncolle/knncolle.hpp"

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
Status<num_dim_, Index_, Float_> initialize(NeighborList<Index_, Float_> nn, const Float_ perp, const Options& options) {
    compute_gaussian_perplexity(nn, perp, options.num_threads);
    symmetrize_matrix(nn);
    return Status<num_dim_, Index_, Float_>(std::move(nn), options);
}

}
/**
 * @endcond
 */

/**
 * Initialize the data structures for t-SNE algorithm, given the nearest neighbors of each observation in the dataset.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the neighbor distances and output embedding.
 *
 * @param neighbors List of indices and distances to nearest neighbors for each observation. 
 * Each observation should have the same number of neighbors, sorted by increasing distance.
 * Each observation should not be included in its own list of neighbors.
 * It is assumed that `neighbors.size()` will fit in an `Index_`.
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
 * Overload of `initialize()` that accepts a neighbor search index and computes the nearest neighbors for each observation,
 * before proceeding with the initialization of the t-SNE algorithm.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Input_ Floating-point type of the input data for the neighbor search.
 * This is only used to define the `knncolle::Prebuilt` type and is otherwise ignored.
 * @tparam Float_ Floating-point type of the neighbor distances and output embedding.
 *
 * @param prebuilt A pre-built neighbor search index for the dataset of interest.
 * @param options Further options.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<std::size_t num_dim_, typename Index_, typename Input_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(const knncolle::Prebuilt<Index_, Input_, Float_>& prebuilt, const Options& options) { 
    const Index_ K = perplexity_to_k<Index_>(options.perplexity); 
    auto neighbors = knncolle::find_nearest_neighbors(prebuilt, K, options.num_threads);
    return internal::initialize<num_dim_>(std::move(neighbors), static_cast<Float_>(options.perplexity), options);
}

/**
 * Overload of `initialize()` that accepts a column-major matrix of coordinates and computes the nearest neighbors for each observation,
 * before proceeding with the initialization of the t-SNE algorithm.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the input data, neighbor distances and output embedding.
 * @tparam Matrix_ Class of the input matrix for the neighbor search.
 * This should be `knncolle::SimpleMatrix` or `knncolle::Matrix`.
 *
 * @param data_dim Number of rows of the matrix at `data`, corresponding to the dimensions of the input dataset.
 * @param num_obs Number of columns of the matrix at `data`, corresponding to the observations of the input dataset.
 * @param[in] data Pointer to an array containing a column-major matrix with `data_dim` rows and `num_obs` columns.
 * @param builder A `knncolle::Builder` instance specifying the nearest-neighbor algorithm to use.
 * @param options Further options.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<std::size_t num_dim_, typename Index_, typename Float_, class Matrix_>
Status<num_dim_, Index_, Float_> initialize(
    const std::size_t data_dim,
    const Index_ num_obs,
    const Float_* const data,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    const Options& options) 
{
    auto index = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(data_dim, num_obs, data));
    return initialize<num_dim_>(*index, options);
}

}

#endif
